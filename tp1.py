####################
# Test (修复后)
####################
test_loss = 0.0
test_loss_ging = 0.0
test_loss_seg = 0.0
test_loss_cur = 0.0
count = 0.0
model.eval()
test_true_cls = []
test_pred_cls = []
test_true_seg = []
test_pred_seg = []
test_label_seg = []
sample_count = 0
with torch.no_grad():
    for data, label, seg in test_loader:
        sample_count += data.shape[0]
        # 修复1: 添加标签的one-hot编码
        label_one_hot = label

        # 修复3: 先转移数据到GPU
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)

        # 在GPU上计算曲率
        AvgAngle_curvature = data[:, :, 7]
        high_cur_num = int(len(AvgAngle_curvature[0]) * args.cur_threshold)
        AvgAngle_idx = torch.topk(AvgAngle_curvature, k=high_cur_num, dim=1)[1]
        idx_base = torch.arange(0, AvgAngle_curvature.size(0), device=device).reshape(-1,
                                                                                      1) * AvgAngle_curvature.size(
            1)
        AvgAngle_idx = (AvgAngle_idx + idx_base).view(-1)

        # 修复2: 确保正确的数据形状
        data = data.permute(0, 2, 1).contiguous()  # [B, 8, N]
        batch_size = data.size()[0]

        seg_pred, ging_pred = model(data, label_one_hot)  # [B, 33, N], [B, 2, N]
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        ging_pred = ging_pred.permute(0, 2, 1).contiguous()
        ging_gold = (seg.view(-1, 1).squeeze() > 0).to(dtype=torch.int64)

        loss_ging = args.w_ging * criterion1(ging_pred.view(-1, 2), ging_gold)
        loss_seg = criterion1(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze())
        loss_cur = args.w_cur * criterion2(seg_pred.view(-1, seg_num_all), seg.view(-1, 1).squeeze(),
                                           AvgAngle_idx)
        loss = loss_ging + loss_seg + loss_cur

        pred = seg_pred.max(dim=2)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_loss_ging += loss_ging.item() * batch_size
        test_loss_cur += loss_cur.item() * batch_size
        test_loss_seg += loss_seg.item() * batch_size
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        print(f"epoch{epoch} the test predict is {pred_np}")
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label[:, 0].type(torch.int32).reshape(-1))
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg)
    outstr = 'Test :: loss: %.6f, test_loss_ging: %.6f, test_loss_seg: %.6f, test_loss_cur: %.6f, ' \
             'test acc: %.6f, test avg acc: %.6f, test iou: %.6f, ' % \
             (test_loss * 1.0 / count,
              test_loss_ging * 1.0 / count,
              test_loss_seg * 1.0 / count,
              test_loss_cur * 1.0 / count,
              test_acc,
              avg_per_class_acc,
              np.mean(test_ious))
io.cprint(outstr)
if np.mean(test_ious) >= best_test_iou:
    best_test_iou = np.mean(test_ious)
    best_test_epoch = epoch
    # torch.save(model.state_dict(), os.path.join(args.save_path, '/%s/models/best_model.t7' % args.exp_name))
    # save_path=os.path.join(args.save_path, '/models/best_model.t7')
    save_path = args.save_path + "/exp/models/best_model2.t7"
    torch.save(model.state_dict(), save_path)
io.cprint("best_test_iou: " + str(best_test_iou) + '; best_test_epoch: ' + str(best_test_epoch))