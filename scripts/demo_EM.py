import tensorflow as tf
import tensorflow_addons as tfa
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('/home/Alexandrite/leekanggeun/CVPR/ISCL/')
from utils.image_tool import *
from utils.parser import parse_args
from models.trainer import Trainer

def ISCL(args, train_clean, train_noisy, test_noisy):
    clean = image_division(image_augmentation(train_clean), patch_size=(64,64))
    noisy = image_division(image_augmentation(train_noisy), patch_size=(64,64))

    # Preprocessing 
    dataset = tf.data.Dataset.from_tensor_slices((clean, noisy)) # If you don't have enough memory, you can use tf.data.Dataset.from_generator
    dataset = dataset.cache().repeat().shuffle(len(train_clean), reshuffle_each_iteration=True).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Training
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE = args.batch_size*strategy.num_replicas_in_sync

    with strategy.scope():
        model = Trainer(args)
        model.compile()
    # if validation data is avaliable.
    #callbacks = []
    #callbacks.append()
    #model.fit(dataset, callbacks=callback, epochs=args.epoch, validation_data=(test_data, test_label), steps_per_epoch=int(len(data)/args.batch_size))

    # if validation data is unavaliable.
    model.fit(dataset, epochs=args.epoch, steps_per_epoch=args.iter) 

    # Testing
    pred = np.zeros(np.shape(test_noisy), dtype=np.float32)
    for i in range(0,len(test_noisy)):
        pred[i] = np.squeeze(model.predict(test_noisy[i:i+1,:,:,np.newaxis])) # noisy: [1, H, W, 1] or [1, H, W, C]
    return pred
def main():
    args = parse_args()
    if args is None:
        exit()
    # Image load
    clean_data = np.array(image_read(args.clean_data), dtype=np.float32) # 
    noisy_data = np.array(image_read(args.noisy_data), dtype=np.float32) # 
    if args.test_data != None:
        test_data = label_read(args.test_data)
    clean_data /= 255
    clean_data = clean_data*2-1
    noisy_data /= 255
    noisy_data = noisy_data*2-1

    # K-fold cross validation
    np.random.seed(2019)
    kf = np.random.permutation(len(clean_data))
    kf = np.reshape(kf, [args.kfold, -1])

    cv_psnr = []
    cv_ssim = []
    for k, test_index in enumerate(kf):
        train_index = np.array(np.setdiff1d(np.array(range(0,len(clean_data))), test_index), dtype=np.int32)
        half = int(len(train_index)/2)
        train_clean = clean_data[train_index]
        train_noisy = noisy_data[train_index]
        test_label = clean_data[test_index]
        test_noisy = noisy_data[test_index]
        
        train_clean = train_clean[:half]
        train_noisy = train_noisy[half:]

        # Testing
        pred = ISCL(args, train_clean, train_noisy, test_noisy)
        test_label = (test_label+1)*0.5*255
        psnr_tot = 0.0
        ssim_tot = 0.0
        for i in range(0,len(test_index)):
            psnr_tot += psnr(test_label[i], pred[i], data_range=255.0)
            ssim_tot += ssim(test_label[i], pred[i], data_range=255.0)
        cv_psnr.append(psnr_tot/len(test_index))
        cv_ssim.append(ssim_tot/len(test_index))
    print("%d-fold cross validation result PSNR: %f, SSIM %f" % (args.kfold, sum(cv_psnr)/4, sum(cv_ssim)/4))

if __name__ == '__main__':
    main()
