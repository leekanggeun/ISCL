import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_addons as tfa
import sys
sys.path.append('/home/Alexandrite/leekanggeun/CVPR/ISCL/')
from models.network import Generator, Discriminator, Extractor
from utils.metrics import PSNR, SSIM

class Trainer(Model):
    def __init__(self, args):
        super(Trainer, self).__init__()
        # Initialization
        self.initializer = 'truncated_normal'
        self.gamma = 0.5
        self.l = 30

        # Models
        self.G = Generator(1, 32, 3, initializer=self.initializer) # Clean image -> Noisy image
        self.F = Generator(1, 32, 3, initializer=self.initializer) # Noisy image -> Clean image
        self.H = Extractor(32, initializer=self.initializer) # Noisy image -> Noise map
        self.DX = Discriminator(32, initializer=self.initializer) # Noisy image discriminator
        self.DY = Discriminator(32, initializer=self.initializer) # Clean image discriminator

        # Optimizers
        self.G_opt = tfa.optimizers.RectifiedAdam(learning_rate=args.lr, beta_1=0.9, warmup_proportion=0.0, total_steps=int(args.iter*args.epoch), min_lr=1e-7)
        self.D_opt = tfa.optimizers.RectifiedAdam(learning_rate=args.lr, beta_1=0.9, warmup_proportion=0.0, total_steps=int(args.iter*args.epoch), min_lr=1e-7)
        self.H_opt = tfa.optimizers.RectifiedAdam(learning_rate=args.lr, beta_1=0.9, warmup_proportion=0.0, total_steps=int(args.iter*args.epoch), min_lr=1e-7)
        self.G_optimizer = tfa.optimizers.SWA(self.G_opt, start_averaging=int(args.iter*args.epoch*0.6), average_period=1)
        self.D_optimizer = tfa.optimizers.Lookahead(self.D_opt, sync_period=6, slow_step_size=0.5)
        self.H_optimizer = tfa.optimizers.Lookahead(self.H_opt, sync_period=6, slow_step_size=0.5)

        # Trackers
        self.gloss_tracker = [tf.keras.metrics.Mean(name="N-gloss"), tf.keras.metrics.Mean(name="C-gloss"), tf.keras.metrics.Mean(name="Cycle"), tf.keras.metrics.Mean(name="Bypass")]
        self.dloss_tracker = [tf.keras.metrics.Mean(name="N-dloss"), tf.keras.metrics.Mean(name="C-dloss"), tf.keras.metrics.Mean(name="Boosting")]
        self.hloss_tracker = [tf.keras.metrics.Mean(name="Pseudo"), tf.keras.metrics.Mean(name="Noise-consistency")]
        self.val_tracker = [PSNR(), SSIM()]

        # Reset tracker
        for tracker in self.gloss_tracker:
            tracker.reset_state()
        for tracker in self.dloss_tracker:
            tracker.reset_state()
        for tracker in self.hloss_tracker:
            tracker.reset_state()
        for tracker in self.val_tracker:
            tracker.reset_state()

    def compile(self, **kwargs):
        self._configure_steps_per_execution(1)
        self._reset_compile_cache()
        self._is_compiled =True
        self.loss = {}

    def call(self, noisy, training=True):
        y_hat = self.F(noisy, training=training)
        y_bar = tf.clip_by_value(noisy-self.H(noisy, training=training), -1.0, 1.0) # We suppose that all images are in range [-1, 1].
        return self.gamma*(y_hat)+(1-self.gamma)*y_bar

    def train_generator(self, clean, noisy):
        with tf.GradientTape(persistent=True) as tape:
            y_hat_i = self.F(noisy, training=True)
            x_hat_j = self.G(clean, training=True)
            #y_bar_i = tf.clip_by_value(noisy-H(noisy, training=False), 1.0, 1.0)
            #x_bar_j = tf.clip_by_value(clean+H(noisy, training=False), 1.0, 1.0)
            y_bar_i = noisy-self.H(noisy, training=False)
            x_bar_j = clean+self.H(noisy, training=False)
            y_hat_j = self.F(x_bar_j, training=True)
            x_tilda_i = self.G(y_hat_i, training=True)
            y_tilda_j = self.F(x_hat_j, training=True)
            
            # Generator loss
            fake_clean = self.DY(y_hat_i, training=False)
            fake_noisy = self.DX(x_hat_j, training=False)
            noisy_gloss = -tf.reduce_mean(fake_noisy)
            clean_gloss = -tf.reduce_mean(fake_clean)
            gloss = noisy_gloss+clean_gloss
            
            # Cycle loss
            cycle_loss = tf.reduce_mean(tf.abs(noisy-x_tilda_i))+tf.reduce_mean(tf.abs(clean-y_tilda_j))

            # Bypass loss
            bypass_loss = tf.reduce_mean(tf.abs(y_hat_i-y_bar_i))+tf.reduce_mean(tf.abs(clean-y_hat_j))
            
            # Nested loss
            nested_loss = cycle_loss+bypass_loss
            # Total loss
            loss = gloss+self.l*nested_loss
        gradient_g = tape.gradient(loss, self.F.trainable_variables+self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(gradient_g, self.F.trainable_variables+self.G.trainable_variables))
        self.gloss_tracker[0].update_state(noisy_gloss)
        self.gloss_tracker[1].update_state(clean_gloss)
        self.gloss_tracker[2].update_state(cycle_loss)
        self.gloss_tracker[3].update_state(bypass_loss)
        return {"N-gloss":self.gloss_tracker[0].result(), 
                "C-gloss":self.gloss_tracker[1].result(), 
                "Cycle":self.gloss_tracker[2].result(), 
                "Bypass":self.gloss_tracker[3].result()}

    def train_discriminator(self, clean, noisy):
        with tf.GradientTape(persistent=True) as tape:
            y_hat_i = self.F(noisy, training=False)
            y_bar_i = noisy-self.H(noisy, training=False)
            x_hat_j = self.G(clean, training=False)
            x_bar_j = clean+self.H(noisy, training=False)
            real_noisy = self.DX(noisy, training=True)
            real_clean = self.DY(clean, training=True)
            fake_noisy = self.DX(x_hat_j, training=True)
            fake_clean = self.DY(y_hat_i, training=True)
            fake_noisy2 = self.DX(x_bar_j, training=True)
            fake_clean2 = self.DY(y_bar_i, training=True)

            # Discriminator loss
            noisy_dloss = tf.reduce_mean(tf.nn.relu(1.0-real_noisy))+tf.reduce_mean(tf.nn.relu(fake_noisy))
            clean_dloss = tf.reduce_mean(tf.nn.relu(1.0-real_clean))+tf.reduce_mean(tf.nn.relu(fake_clean))
            
            # Boosting loss
            bst_loss = tf.reduce_mean(tf.nn.relu(fake_noisy2))+tf.reduce_mean(tf.nn.relu(fake_clean2))

            # Total loss
            loss = noisy_dloss+clean_dloss+bst_loss
        gradient_d = tape.gradient(loss, self.DX.trainable_variables+self.DY.trainable_variables)
        self.D_optimizer.apply_gradients(zip(gradient_d, self.DX.trainable_variables+self.DY.trainable_variables))
        self.dloss_tracker[0].update_state(noisy_dloss)
        self.dloss_tracker[1].update_state(clean_dloss)
        self.dloss_tracker[2].update_state(bst_loss)
        return {"N-dloss":self.dloss_tracker[0].result(),
                "C-dloss":self.dloss_tracker[1].result(),
                "Boosting":self.dloss_tracker[2].result()}
    
    def train_extractor(self, clean, noisy):
        with tf.GradientTape(persistent=True) as tape:
            n_hat_i = self.H(noisy, training=True)
            n_bar_i = noisy-self.F(noisy, training=False)
            x_hat_j = self.G(clean, training=False)
            n_tilda_j = self.H(x_hat_j, training=True)
            # Pseudo noise loss
            pseudo_loss = tf.reduce_mean(tf.abs(n_hat_i-n_bar_i))
            noise_consistency = tf.reduce_mean(tf.abs(x_hat_j-clean-n_tilda_j))
            loss = pseudo_loss+noise_consistency
        gradient_h = tape.gradient(loss, self.H.trainable_variables)
        self.H_optimizer.apply_gradients(zip(gradient_h, self.H.trainable_variables))
        self.hloss_tracker[0].update_state(pseudo_loss)
        self.hloss_tracker[1].update_state(noise_consistency)
        return {"Pseudo":self.hloss_tracker[0].result(),
                "Noise-consistency":self.hloss_tracker[1].result()}

    def train_step(self, data):
        clean, noisy = data
        dloss = self.train_discriminator(clean, noisy)
        gloss = self.train_generator(clean, noisy)
        hloss = self.train_extractor(clean, noisy)
        return {**gloss, **dloss, **hloss}
    
    def predict(self, noisy):
        y_hat = self.F(noisy, training=False)
        y_bar = noisy-self.H(noisy, training=False)
        pred = tf.clip_by_value(self.gamma*(y_hat)+(1-self.gamma)*y_bar, -1.0, 1.0)
        pred = (pred+1)*0.5*255
        return pred

    def test_step(self, data):
        # Label range [0,255], prediction range [-1, 1]
        label, data = data
        predict = tf.squeeze(self.predict(data), 0)
        label = tf.squeeze(label, 0)
        self.val_tracker[0].update_state(label, predict)
        self.val_tracker[1].update_state(label, predict)
        return {"PSNR":self.val_tracker[0].result(), 
                "SSIM":self.val_tracker[1].result()}

        