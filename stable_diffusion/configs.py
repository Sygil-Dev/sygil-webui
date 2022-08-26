from omegaconf.omegaconf import OmegaConf

"""
Dealing with packaging files is way too annoying.
"""

stable_diffusion_v1 = OmegaConf.create("""
model:
  base_learning_rate: 1.0e-04
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false   # Note: different from the one we trained before
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    scheduler_config: # 10000 warmup steps
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [ 10000 ]
        cycle_lengths: [ 10000000000000 ] # incredibly large number to prevent corner cases
        f_start: [ 1.e-6 ]
        f_max: [ 1. ]
        f_min: [ 1. ]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
""")

stable_diffusion_v1_optimized = OmegaConf.create("""
modelUNet:
  base_learning_rate: 1.0e-04
  target: stable_diffusion.diffusion.ddpm.UNet
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false # Note: different from the one we trained before
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    unetConfigEncode:
      target: stable_diffusion.diffusion.openaimodelSplit.UNetModelEncode
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2, 1]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    unetConfigDecode:
      target: stable_diffusion.diffusion.openaimodelSplit.UNetModelDecode
      params:
        image_size: 32 # unused
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2, 1]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

modelFirstStage:
  target: stable_diffusion.diffusion.ddpm.FirstStage
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false # Note: different from the one we trained before
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False
    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
            - 1
            - 2
            - 4
            - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

modelCondStage:
  target: stable_diffusion.diffusion.ddpm.CondStage
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: "jpg"
    cond_stage_key: "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: false # Note: different from the one we trained before
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False
    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
""")

autoencoder_kl_8x8x64 = OmegaConf.create("""
model:
  base_learning_rate: 4.5e-6
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    monitor: "val/rec_loss"
    embed_dim: 64
    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 50001
        kl_weight: 0.000001
        disc_weight: 0.5

    ddconfig:
      double_z: True
      z_channels: 64
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [ 1,1,2,2,4,4]  # num_down = len(ch_mult)-1
      num_res_blocks: 2
      attn_resolutions: [16,8]
      dropout: 0.0

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 12
    wrap: True
    train:
      target: ldm.data.imagenet.ImageNetSRTrain
      params:
        size: 256
        degradation: pil_nearest
    validation:
      target: ldm.data.imagenet.ImageNetSRValidation
      params:
        size: 256
        degradation: pil_nearest

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 1000
        max_images: 8
        increase_log_steps: True

  trainer:
    benchmark: True
    accumulate_grad_batches: 2
""")

autoencoder_kl_16x16x16 = OmegaConf.create("""
model:
  base_learning_rate: 4.5e-6
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    monitor: "val/rec_loss"
    embed_dim: 16
    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 50001
        kl_weight: 0.000001
        disc_weight: 0.5

    ddconfig:
      double_z: True
      z_channels: 16
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [ 1,1,2,2,4]  # num_down = len(ch_mult)-1
      num_res_blocks: 2
      attn_resolutions: [16]
      dropout: 0.0


data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 12
    wrap: True
    train:
      target: ldm.data.imagenet.ImageNetSRTrain
      params:
        size: 256
        degradation: pil_nearest
    validation:
      target: ldm.data.imagenet.ImageNetSRValidation
      params:
        size: 256
        degradation: pil_nearest

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 1000
        max_images: 8
        increase_log_steps: True

  trainer:
    benchmark: True
    accumulate_grad_batches: 2
""")

autoencoder_kl_32x32x4 = OmegaConf.create("""
model:
  base_learning_rate: 4.5e-6
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    monitor: "val/rec_loss"
    embed_dim: 4
    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 50001
        kl_weight: 0.000001
        disc_weight: 0.5

    ddconfig:
      double_z: True
      z_channels: 4
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [ 1,2,4,4 ]  # num_down = len(ch_mult)-1
      num_res_blocks: 2
      attn_resolutions: [ ]
      dropout: 0.0

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 12
    wrap: True
    train:
      target: ldm.data.imagenet.ImageNetSRTrain
      params:
        size: 256
        degradation: pil_nearest
    validation:
      target: ldm.data.imagenet.ImageNetSRValidation
      params:
        size: 256
        degradation: pil_nearest

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 1000
        max_images: 8
        increase_log_steps: True

  trainer:
    benchmark: True
    accumulate_grad_batches: 2
""")

autoencoder_kl_64x64x3 = OmegaConf.create("""
model:
  base_learning_rate: 4.5e-6
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    monitor: "val/rec_loss"
    embed_dim: 3
    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 50001
        kl_weight: 0.000001
        disc_weight: 0.5

    ddconfig:
      double_z: True
      z_channels: 3
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [ 1,2,4 ]  # num_down = len(ch_mult)-1
      num_res_blocks: 2
      attn_resolutions: [ ]
      dropout: 0.0


data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 12
    wrap: True
    train:
      target: ldm.data.imagenet.ImageNetSRTrain
      params:
        size: 256
        degradation: pil_nearest
    validation:
      target: ldm.data.imagenet.ImageNetSRValidation
      params:
        size: 256
        degradation: pil_nearest

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 1000
        max_images: 8
        increase_log_steps: True

  trainer:
    benchmark: True
    accumulate_grad_batches: 2
""")

bsr_sr = OmegaConf.create("""
model:
  base_learning_rate: 1.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0155
    log_every_t: 100
    timesteps: 1000
    loss_type: l2
    first_stage_key: image
    cond_stage_key: LR_image
    image_size: 64
    channels: 3
    concat_mode: true
    cond_stage_trainable: false
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 6
        out_channels: 3
        model_channels: 160
        attention_resolutions:
        - 16
        - 8
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 2
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        monitor: val/rec_loss
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: torch.nn.Identity
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 64
    wrap: false
    num_workers: 12
    train:
      target: ldm.data.openimages.SuperresOpenImagesAdvancedTrain
      params:
        size: 256
        degradation: bsrgan_light
        downscale_f: 4
        min_crop_f: 0.5
        max_crop_f: 1.0
        random_crop: true
    validation:
      target: ldm.data.openimages.SuperresOpenImagesAdvancedValidation
      params:
        size: 256
        degradation: bsrgan_light
        downscale_f: 4
        min_crop_f: 0.5
        max_crop_f: 1.0
        random_crop: true
""")

celeba_256 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 64
    channels: 3
    cond_stage_trainable: false
    concat_mode: false
    monitor: val/loss
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config: __is_unconditional__
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 48
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.faceshq.CelebAHQTrain
      params:
        size: 256
    validation:
      target: ldm.data.faceshq.CelebAHQValidation
      params:
        size: 256
""")

celebahq_ldm_vq_4 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    image_size: 64
    channels: 3
    monitor: val/loss_simple_ema

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions:
        # note: this isn\t actually the resolution but
        # the downsampling factor, i.e. this corresnponds to
        # attention on spatial resolution 8,16,32, as the
        # spatial reolution of the latents is 64 for f4
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ckpt_path: models/first_stage_models/vq-f4/model.ckpt
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config: __is_unconditional__
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 48
    num_workers: 5
    wrap: false
    train:
      target: taming.data.faceshq.CelebAHQTrain
      params:
        size: 256
    validation:
      target: taming.data.faceshq.CelebAHQValidation
      params:
        size: 256


lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
""")

cin256 = OmegaConf.create("""
model:
  base_learning_rate: 1.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 32
    channels: 4
    cond_stage_trainable: true
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 256
        attention_resolutions:
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 4
        num_head_channels: 32
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 512
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 4
        n_embed: 16384
        ddconfig:
          double_z: false
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions:
          - 32
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.ClassEmbedder
      params:
        embed_dim: 512
        key: class_label
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 64
    num_workers: 12
    wrap: false
    train:
      target: ldm.data.imagenet.ImageNetTrain
      params:
        config:
          size: 256
    validation:
      target: ldm.data.imagenet.ImageNetValidation
      params:
        config:
          size: 256
""")

cin256_v2 = OmegaConf.create("""
model:
  base_learning_rate: 0.0001
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 64
    channels: 3
    cond_stage_trainable: true
    conditioning_key: crossattn
    monitor: val/loss
    use_ema: False
    
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 192
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 5
        num_heads: 1
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 512
    
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    
    cond_stage_config:
      target: ldm.modules.encoders.modules.ClassEmbedder
      params:
        n_classes: 1001
        embed_dim: 512
        key: class_label
""")

cin_ldm_vq_f8 = OmegaConf.create("""
model:
  base_learning_rate: 1.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 32
    channels: 4
    cond_stage_trainable: true
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 256
        attention_resolutions:
        #note: this isn\t actually the resolution but
        # the downsampling factor, i.e. this corresnponds to
        # attention on spatial resolution 8,16,32, as the
        # spatial reolution of the latents is 32 for f8
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 4
        num_head_channels: 32
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 512
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 4
        n_embed: 16384
        ckpt_path: configs/first_stage_models/vq-f8/model.yaml
        ddconfig:
          double_z: false
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions:
          - 32
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.ClassEmbedder
      params:
        embed_dim: 512
        key: class_label
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 64
    num_workers: 12
    wrap: false
    train:
      target: ldm.data.imagenet.ImageNetTrain
      params:
        config:
          size: 256
    validation:
      target: ldm.data.imagenet.ImageNetValidation
      params:
        config:
          size: 256


lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
""")

ffhq256 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 64
    channels: 3
    cond_stage_trainable: false
    concat_mode: false
    monitor: val/loss
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config: __is_unconditional__
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 42
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.faceshq.FFHQTrain
      params:
        size: 256
    validation:
      target: ldm.data.faceshq.FFHQValidation
      params:
        size: 256
""")

ffhq_ldm_vq_4 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    image_size: 64
    channels: 3
    monitor: val/loss_simple_ema
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions:
        # note: this isn\t actually the resolution but
        # the downsampling factor, i.e. this corresnponds to
        # attention on spatial resolution 8,16,32, as the
        # spatial reolution of the latents is 64 for f4
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ckpt_path: configs/first_stage_models/vq-f4/model.yaml
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config: __is_unconditional__
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 42
    num_workers: 5
    wrap: false
    train:
      target: taming.data.faceshq.FFHQTrain
      params:
        size: 256
    validation:
      target: taming.data.faceshq.FFHQValidation
      params:
        size: 256


lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
""")

inpainting_big = OmegaConf.create("""
model:
  base_learning_rate: 1.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0205
    log_every_t: 100
    timesteps: 1000
    loss_type: l1
    first_stage_key: image
    cond_stage_key: masked_image
    image_size: 64
    channels: 3
    concat_mode: true
    monitor: val/loss
    scheduler_config:
      target: ldm.lr_scheduler.LambdaWarmUpCosineScheduler
      params:
        verbosity_interval: 0
        warm_up_steps: 1000
        max_decay_steps: 50000
        lr_start: 0.001
        lr_max: 0.1
        lr_min: 0.0001
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 7
        out_channels: 3
        model_channels: 256
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_heads: 8
        resblock_updown: true
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        monitor: val/rec_loss
        ddconfig:
          attn_type: none
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: ldm.modules.losses.contperceptual.DummyLoss
    cond_stage_config: __is_first_stage__
""")

layout2img_openimages256 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0205
    log_every_t: 100
    timesteps: 1000
    loss_type: l1
    first_stage_key: image
    cond_stage_key: coordinates_bbox
    image_size: 64
    channels: 3
    conditioning_key: crossattn
    cond_stage_trainable: true
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 128
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
        use_spatial_transformer: true
        transformer_depth: 3
        context_dim: 512
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        monitor: val/rec_loss
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.BERTEmbedder
      params:
        n_embed: 512
        n_layer: 16
        vocab_size: 8192
        max_seq_len: 92
        use_tokenizer: false
    monitor: val/loss_simple_ema
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 24
    wrap: false
    num_workers: 10
    train:
      target: ldm.data.openimages.OpenImagesBBoxTrain
      params:
        size: 256
    validation:
      target: ldm.data.openimages.OpenImagesBBoxValidation
      params:
        size: 256
""")

lsun_beds256 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: class_label
    image_size: 64
    channels: 3
    cond_stage_trainable: false
    concat_mode: false
    monitor: val/loss
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config: __is_unconditional__
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 48
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.lsun.LSUNBedroomsTrain
      params:
        size: 256
    validation:
      target: ldm.data.lsun.LSUNBedroomsValidation
      params:
        size: 256
""")

lsun_bedrooms_ldm_vq_4 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    image_size: 64
    channels: 3
    monitor: val/loss_simple_ema
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 224
        attention_resolutions:
        # note: this isn\t actually the resolution but
        # the downsampling factor, i.e. this corresnponds to
        # attention on spatial resolution 8,16,32, as the
        # spatial reolution of the latents is 64 for f4
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        num_head_channels: 32
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        ckpt_path: configs/first_stage_models/vq-f4/model.yaml
        embed_dim: 3
        n_embed: 8192
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config: __is_unconditional__
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 48
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.lsun.LSUNBedroomsTrain
      params:
        size: 256
    validation:
      target: ldm.data.lsun.LSUNBedroomsValidation
      params:
        size: 256


lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
""")

lsun_churches256 = OmegaConf.create("""
model:
  base_learning_rate: 5.0e-05
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0155
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    loss_type: l1
    first_stage_key: image
    cond_stage_key: image
    image_size: 32
    channels: 4
    cond_stage_trainable: false
    concat_mode: false
    scale_by_std: true
    monitor: val/loss_simple_ema
    scheduler_config:
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps:
        - 10000
        cycle_lengths:
        - 10000000000000
        f_start:
        - 1.0e-06
        f_max:
        - 1.0
        f_min:
        - 1.0
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 192
        attention_resolutions:
        - 1
        - 2
        - 4
        - 8
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 2
        - 4
        - 4
        num_heads: 8
        use_scale_shift_norm: true
        resblock_updown: true
    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config: '__is_unconditional__'

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 96
    num_workers: 5
    wrap: false
    train:
      target: ldm.data.lsun.LSUNChurchesTrain
      params:
        size: 256
    validation:
      target: ldm.data.lsun.LSUNChurchesValidation
      params:
        size: 256
""")

lsun_churches_ldm_kl_8 = OmegaConf.create("""
model:
  base_learning_rate: 5.0e-5   # set to target_lr by starting main.py with '--scale_lr False'
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0155
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    loss_type: l1
    first_stage_key: "image"
    cond_stage_key: "image"
    image_size: 32
    channels: 4
    cond_stage_trainable: False
    concat_mode: False
    scale_by_std: True
    monitor: 'val/loss_simple_ema'

    scheduler_config: # 10000 warmup steps
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [10000]
        cycle_lengths: [10000000000000]
        f_start: [1.e-6]
        f_max: [1.]
        f_min: [ 1.]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 192
        attention_resolutions: [ 1, 2, 4, 8 ]   # 32, 16, 8, 4
        num_res_blocks: 2
        channel_mult: [ 1,2,2,4,4 ]  # 32, 16, 8, 4, 2
        num_heads: 8
        use_scale_shift_norm: True
        resblock_updown: True

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: "val/rec_loss"
        ckpt_path: "models/first_stage_models/kl-f8/model.ckpt"
        ddconfig:
          double_z: True
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [ 1,2,4,4 ]  # num_down = len(ch_mult)-1
          num_res_blocks: 2
          attn_resolutions: [ ]
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config: "__is_unconditional__"

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 96
    num_workers: 5
    wrap: False
    train:
      target: ldm.data.lsun.LSUNChurchesTrain
      params:
        size: 256
    validation:
      target: ldm.data.lsun.LSUNChurchesValidation
      params:
        size: 256

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: False


  trainer:
    benchmark: True
""")

semantic_synthesis256 = OmegaConf.create("""
model:
  base_learning_rate: 1.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0205
    log_every_t: 100
    timesteps: 1000
    loss_type: l1
    first_stage_key: image
    cond_stage_key: segmentation
    image_size: 64
    channels: 3
    concat_mode: true
    cond_stage_trainable: true
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 6
        out_channels: 3
        model_channels: 128
        attention_resolutions:
        - 32
        - 16
        - 8
        num_res_blocks: 2
        channel_mult:
        - 1
        - 4
        - 8
        num_heads: 8
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.SpatialRescaler
      params:
        n_stages: 2
        in_channels: 182
        out_channels: 3
""")

semantic_synthesis512 = OmegaConf.create("""
model:
  base_learning_rate: 1.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0205
    log_every_t: 100
    timesteps: 1000
    loss_type: l1
    first_stage_key: image
    cond_stage_key: segmentation
    image_size: 128
    channels: 3
    concat_mode: true
    cond_stage_trainable: true
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 128
        in_channels: 6
        out_channels: 3
        model_channels: 128
        attention_resolutions:
        - 32
        - 16
        - 8
        num_res_blocks: 2
        channel_mult:
        - 1
        - 4
        - 8
        num_heads: 8
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        monitor: val/rec_loss
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.SpatialRescaler
      params:
        n_stages: 2
        in_channels: 182
        out_channels: 3
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 8
    wrap: false
    num_workers: 10
    train:
      target: ldm.data.landscapes.RFWTrain
      params:
        size: 768
        crop_size: 512
        segmentation_to_float32: true
    validation:
      target: ldm.data.landscapes.RFWValidation
      params:
        size: 768
        crop_size: 512
        segmentation_to_float32: true
""")

text2img256 = OmegaConf.create("""
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0195
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: caption
    image_size: 64
    channels: 3
    cond_stage_trainable: true
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 3
        out_channels: 3
        model_channels: 192
        attention_resolutions:
        - 8
        - 4
        - 2
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 5
        num_head_channels: 32
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 640
    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: ldm.modules.encoders.modules.BERTEmbedder
      params:
        n_embed: 640
        n_layer: 32
data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 28
    num_workers: 10
    wrap: false
    train:
      target: ldm.data.previews.pytorch_dataset.PreviewsTrain
      params:
        size: 256
    validation:
      target: ldm.data.previews.pytorch_dataset.PreviewsValidation
      params:
        size: 256
""")

txt2img_1p4b_eval = OmegaConf.create("""
model:
  base_learning_rate: 5.0e-05
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.012
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: image
    cond_stage_key: caption
    image_size: 32
    channels: 4
    cond_stage_trainable: true
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions:
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 4
        - 4
        num_heads: 8
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 1280
        use_checkpoint: true
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.BERTEmbedder
      params:
        n_embed: 1280
        n_layer: 32
""")

retrieval_augmented_diffusion_768x768 = OmegaConf.create("""
model:
  base_learning_rate: 0.0001
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.015
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_key: jpg
    cond_stage_key: nix
    image_size: 48
    channels: 16
    cond_stage_trainable: false
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_by_std: false
    scale_factor: 0.22765929
    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 48
        in_channels: 16
        out_channels: 16
        model_channels: 448
        attention_resolutions:
        - 4
        - 2
        - 1
        num_res_blocks: 2
        channel_mult:
        - 1
        - 2
        - 3
        - 4
        use_scale_shift_norm: false
        resblock_updown: false
        num_head_channels: 32
        use_spatial_transformer: true
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: true
    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        monitor: val/rec_loss
        embed_dim: 16
        ddconfig:
          double_z: true
          z_channels: 16
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 1
          - 2
          - 2
          - 4
          num_res_blocks: 2
          attn_resolutions:
          - 16
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity
    cond_stage_config:
      target: torch.nn.Identity
""")
