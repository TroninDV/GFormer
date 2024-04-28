import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import Utils.TimeLogger as logger
from DataHandler import DataHandler
from Model import GTLayer, LocalGraph, Model, RandomMaskSubgraphs
from Params import args
from torch.utils.tensorboard import SummaryWriter
from Utils.TimeLogger import log
from Utils.Utils import *
from Utils.Utils import contrast

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

        self.writer = SummaryWriter(comment=args.experiment)

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        bestRes = None
        result = []
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            
            for metric_name, metric_value in reses.items():
                self.writer.add_scalar(f"{metric_name}/train", metric_value, ep)

            if tstFlag:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
                result.append(reses)

                for metric_name, metric_value in reses.items():
                    self.writer.add_scalar(f"{metric_name}/test", metric_value, ep)

                bestRes = reses if bestRes is None or reses['Recall'] > bestRes['Recall'] else bestRes
            print()
        reses = self.testEpoch()
        result.append(reses)
        torch.save(result, "Saeg_result.pkl")
        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('Best Result', args.epoch, bestRes, True))
        self.saveHistory()
        self.writer.close()

    def prepareModel(self):
        self.gtLayer = GTLayer().cuda()
        self.model = Model(self.gtLayer).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(args.user, args.item)
        self.sampler = LocalGraph(self.gtLayer)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0

        loss_parts = defaultdict(int)

        steps = trnLoader.dataset.__len__() // args.batch
        self.handler.preSelect_anchor_set()
        for i, tem in enumerate(trnLoader):
            if i % args.fixSteps == 0:
                att_edge, add_adj = self.sampler(self.handler.torchBiAdj, self.model.getEgoEmbeds(),
                                                 self.handler)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            usrEmbeds, itmEmbeds, cList, subLst = self.model(self.handler, False, sub, cmp,  encoderAdj,
                                                                           decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]

            usrEmbeds2 = subLst[:args.user]
            itmEmbeds2 = subLst[args.user:]
            ancEmbeds2 = usrEmbeds2[ancs]
            posEmbeds2 = itmEmbeds2[poss]

            # bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            bprLos = F.binary_cross_entropy(torch.ones_like(ancs), ancEmbeds @ posEmbeds.T)
            #
            scoreDiff = pairPredict(ancEmbeds2, posEmbeds2, negEmbeds)
            bprLoss2 = - (scoreDiff).sigmoid().log().sum() / args.batch

            regLoss = calcRegLoss(self.model) * args.reg

            universalityLossDistinct = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg

            universalityLossCommon = contrast(
                ancs,
                usrEmbeds,
                itmEmbeds)

            contrastNCELoss = args.ctra*contrastNCE(ancs, subLst, cList)

            contrastLoss = universalityLossDistinct + universalityLossCommon + contrastNCELoss
            loss = bprLoss + regLoss + contrastLoss + args.b2*bprLoss2

            loss_parts['pos_embed_loss'] += bprLoss.item() 
            loss_parts['contrast_embed_loss'] += bprLoss2.item() 
            loss_parts['reg_loss'] += regLoss.item() 
            loss_parts['universality_loss_distinct'] += universalityLossDistinct.item() 
            loss_parts['universality_loss_common'] += universalityLossCommon.item() 
            loss_parts['contrast_nce_loss'] += contrastNCELoss.item() 


            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            log('Step %d/%d: loss = %.3f, regLoss = %.3f, clLoss = %.3f        ' % (
                i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps

        for metric_name, metric_value in loss_parts.items():
            ret[metric_name] = metric_value / steps

        return ret

    def testEpoch(self):
        from collections import defaultdict

        tstLoader = self.handler.tstLoader

        epRecall = defaultdict(int)
        epNdcg = defaultdict(int)
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        k_values = [1, 5, 10, 20, 100]

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds, _, _ = self.model(self.handler, True, self.handler.torchBiAdj, self.handler.torchBiAdj,
                                                          self.handler.torchBiAdj)

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            for k in k_values:
                _, topLocs = t.topk(allPreds, k)
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall[k] += recall
            epNdcg[k] += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
        ret = dict()
        for k in k_values:
            ret[f'Recall/@{k}'] = epRecall[k] / num
            ret[f'NDCG/@{k}'] = epNdcg[k] / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, './Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load('./Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('./History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    logger.saveDefault = True

    log('Start')
    if t.cuda.is_available():
        print("using cuda")
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
