from sklearn.metrics import average_precision_score, accuracy_score, f1_score, normalized_mutual_info_score,\
    adjusted_mutual_info_score, adjusted_rand_score



def get_metrics(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    nmi = normalized_mutual_info_score(preds, labels)
    ari = adjusted_rand_score(preds, labels)
    ami = adjusted_mutual_info_score(preds, labels)
    return accuracy, f1, nmi, ari, ami

