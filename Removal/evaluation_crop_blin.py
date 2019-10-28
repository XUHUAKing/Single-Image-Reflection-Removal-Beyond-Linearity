'''
eval script for reflectSuppress results
steps:
    1. download ./crop_npy
    2. run python npy2png.py --> convert .npy to .png
    3. run test.m in matlab --> generate results
    4. run python res2folder.py --> reorganize results for evaluation
    5. cd results/ && rm *.png --> remove .png files after reorganize
    6. run evaluation_crop.py (this script) --> evaluation
'''
from imageio import imread, imsave
from glob import glob
from skimage.measure import compare_psnr, compare_ssim
import numpy as np
import cv2

def prepare_data_npy(data_path='./crop_npy/'):
    train_items, val_items = [], []
    folders1 = glob(data_path+'/*')
#    print(folders1)
    folders2 = []
    for folder1 in folders1:
        folders2 = folders2 + glob(folder1+'/Indoor/*') + glob(folder1+'/Outdoor/*')
#    print(folders2)
    folders2.sort()
    for folder2 in folders2[1::5] + folders2[2::5]+folders2[3::5]+folders2[4::5]:
        folder = folder2
        imgs = glob(folder + '/*.npy')
        imgs.sort()
        print(folder, len(imgs))
        for idx in range(len(imgs)//2):
            tmp_M = imgs[2*idx+1]
            tmp_R = imgs[2*idx]
            train_items.append([tmp_M,tmp_R])
            print(tmp_R, tmp_M)

    for folder2 in folders2[::5]:
        folder = folder2
        imgs = glob(folder + '/*.npy')
        imgs.sort()
        print(folder, len(imgs))
        for idx in range(len(imgs)//2):
            tmp_M = imgs[2*idx+1]
            tmp_R = imgs[2*idx]
            val_items.append([tmp_M,tmp_R])
            print(tmp_R, tmp_M)


    return train_items, val_items[::3]

def prepare_results(data_path='./results/'):
    pred_images = []
    folders1 = glob(data_path+'/*')
#    print(folders1)
    folders2 = []
    for folder1 in folders1:
        folders2 = folders2 + glob(folder1+'/Indoor/*') + glob(folder1+'/Outdoor/*')
#    print(folders2)
    folders2.sort()
    for folder2 in folders2:
        folder = folder2
        imgs = glob(folder + '/*.png')
        imgs.sort()
        print(folder, len(imgs))
        pred_images.extend(imgs)
    return pred_images

train_names, val_names = prepare_data_npy()
print('Data load succeed!')
print(len(train_names), len(val_names))
num_train, num_test = len(train_names), len(val_names)
def prepare_item(item):
    M_name, R_name = item
    tmp_M = np.load(M_name)
    tmp_R = np.load(R_name)
    tmp_M =0.5* np.load(M_name)[:,:,:,[4,4,4]]
    tmp_R =0.5* np.load(R_name)[:,:,:,[4,4,4]]
    #tmp_M = np.load(M_name)[:,:,:,[4,4,4]]
    #tmp_R = np.load(R_name)[:,:,:,[4,4,4]]
    print("M_mean", np.mean(tmp_M), np.max(tmp_M),tmp_M.shape)
    tmp_T = tmp_M - tmp_R
    tmp_T[tmp_T>1] = 1
    tmp_T[tmp_T<0] = 0
    return np.power(tmp_M,1/2.2), np.power(tmp_T,1/2.2), np.power(tmp_R,1/2.2)
#    return np.power(tmp_M,1), np.power(tmp_T,1), np.power(tmp_R,1)

# evaluate and pick best result for each image
pred_images_defocus = prepare_results('./results_defocus')
pred_images_focus = prepare_results('./results_focus')
pred_images_ghost = prepare_results('./results_ghost')
pred_images_defocus.sort()
pred_images_focus.sort()
pred_images_ghost.sort()
num_image = min([len(pred_images_defocus), len(pred_images_focus), len(pred_images_ghost)])
all_ssim, all_psnr = 0,0
for idx in range(num_image):
    all, gt, R = prepare_item(val_names[idx])
    pred_defocus = cv2.imread(pred_images_defocus[idx],-1)/65535.
    pred_focus = cv2.imread(pred_images_focus[idx],-1)/65535.
    pred_ghost = cv2.imread(pred_images_ghost[idx],-1)/65535.
    # duplicate along RGB
    pred_defocus = np.tile(pred_defocus[:,:,np.newaxis],[1,1,3])
    pred_focus = np.tile(pred_focus[:,:,np.newaxis],[1,1,3])
    pred_ghost = np.tile(pred_ghost[:,:,np.newaxis],[1,1,3])

    gt = gt[0,:,:,:]
    all = all[0,:,:,:]

    h1,w1 = all.shape[:2]
    h2,w2 = pred_defocus.shape[:2]
    h, w = min(h1,h2), min(w1, w2)
    h = h // 32 * 32
    w = w // 32 * 32
    pred_defocus = pred_defocus[:h,:w,:]
    pred_focus = pred_focus[:h,:w,:]
    pred_ghost = pred_ghost[:h,:w,:]
    gt = gt[:h,:w,:]
    all =all[:h,:w,:]
    print("mean is {}, pred_defocus mean is {}".format(np.mean(gt), np.mean(pred_defocus)))
    print("mean is {}, pred_focus mean is {}".format(np.mean(gt), np.mean(pred_focus)))
    print("mean is {}, pred_ghost mean is {}".format(np.mean(gt), np.mean(pred_ghost)))
    '''with gamma correction inside model'''
    psnrL = []
    psnrL.append(compare_psnr(np.power(np.mean(pred_defocus,axis=2),2.2), np.power(gt[:,:,0],2.2),1))
    psnrL.append(compare_psnr(np.power(np.mean(pred_focus,axis=2),2.2), np.power(gt[:,:,0],2.2),1))
    psnrL.append(compare_psnr(np.power(np.mean(pred_ghost,axis=2),2.2), np.power(gt[:,:,0],2.2),1))
    # pick the best one
    psnr = max(psnrL)
    Rtype = ['defocus','focus','ghost'][np.argmax(psnrL)]
    if Rtype == 'defocus':
        pred = pred_defocus
    elif Rtype == 'focus':
        pred = pred_focus
    else:
        pred = pred_ghost
    ssim = compare_ssim(np.power(np.mean(pred,axis=2),2.2), np.power(gt[:,:,0],2.2))
    all_ssim += ssim
    all_psnr += psnr
    imsave('evaluation/IMG_%04d.jpg'%idx,np.concatenate([all, pred, gt],axis=1))
    print(ssim, psnr)
print(all_ssim*1.0/(idx+1), all_psnr*1.0/(idx+1))
