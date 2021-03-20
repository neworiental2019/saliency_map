import matplotlib.pyplot as plt
import numpy as np
import torch


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def predictions_to_class_info(pred, class_labels):
    pred = softmax(pred)
    class_id = np.argmax(pred)
    class_prob = pred[class_id]
    #class_labels = np.loadtxt(open(label_file), dtype=object, delimiter='\n')
    class_labels = str(class_labels)
    return class_id, class_labels[class_id], pred[class_id]


def compute_saliency_map(model, tensor_image, k_size, ref_class_id, ref_class_prob, thr=0.0):
    assert k_size >= 3 and k_size % 2 == 1

    saliency_values = []
    ch, rows, cols = tensor_image.shape
    h_size = int(k_size / 2)

    for u in range(h_size, rows, h_size):
        for v in range(h_size, cols, h_size):
             
            #print('u',u,'v',v,'h_size',h_size,u-h_size,u+h_size,v-h_size,v+h_size)
            masked_image = tensor_image.clone()
            mask_color = torch.mean(masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size])
            masked_image[:, u-h_size:u+h_size, v-h_size:v+h_size] = mask_color
            #print(type(masked_image),masked_image.shape)
            #show = torch.reshape(masked_image,(224,224,3))
            #show = show.cpu().detach().numpy()
            #plt.imshow(show)
            #plt.show()
            img = masked_image.unsqueeze(0)
            pred = model(torch.autograd.Variable(img)) # 1000 scores on gpu
            pred = pred.data.cpu().numpy().squeeze() # ... on cpu
            pred_class_id = np.argmax(pred)
            pred = 0.0 if ref_class_id != pred_class_id else softmax(pred)[pred_class_id]
            pred_err = ref_class_prob - pred
            if np.abs(pred_err) < thr:
                pred_err = 0.0
            saliency_values.append(pred_err)
    
    size = int(np.sqrt(len(saliency_values)))
    saliency_map = np.array(saliency_values, dtype=np.float32)
    saliency_map = saliency_map.reshape((size, size))
    
    #plt.imshow(saliency_map)
    #plt.show()
    return saliency_map


def get_feature_maps(model, tensor_image, layer_id=None, has_cuda=False):
    layers = list(model.features.children())
    if layer_id is not None:
        if layer_id > 0:
            layer_id += 1
        layers = layers[:layer_id]
    feature_extractor = torch.nn.Sequential(*layers)
    if has_cuda:
        feature_extractor.cuda()
    feats = feature_extractor(torch.autograd.Variable(tensor_image.unsqueeze(0)))
    feats = feats.data.cpu().squeeze_().numpy()
    return feats


def show_feature_maps(feats):
    import matplotlib.pyplot as plt

    nfts = feats.shape[0]
    h_size = int(np.ceil(np.sqrt(nfts)))

    plt.figure()
    for i in range(1, nfts + 1):
        plt.subplot(h_size, h_size, i)
        plt.imshow(feats[i-1])
    plt.show()


def compute_feature_correlation(feats):
    nfts = feats.shape[0]
    diff_mat = np.zeros((nfts, nfts), dtype=np.float32)
    for i in range(nfts):
        for j in range(nfts):
            if i != j:
                diff_mat[i,j] = np.mean(np.abs(feats[i] - feats[j]))
    return diff_mat


