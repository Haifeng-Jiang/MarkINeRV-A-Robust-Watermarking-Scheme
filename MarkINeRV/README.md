# MarkINeRV

This is the official code for "MarkINeRV: A Robust Watermarking Scheme for Neural Representation for Videos Based on Invertible Neural Networks"

# Running the code

This code requires Python 3. 

You can find the pretrained models at `model.pt`.

---


To train a NeRV:
```
python train_nerv_all.py
```
After training for 300 iterations (~2 hours on a single 2080 Ti), you can find the following  at `Encoder_0.31M_Decoder_1.49M_Total_1.5M`.

---
To render video frame from different timestamps
```
python efficient_nvloader.py 
```

---


To train a INN (Implement watermark embedding and extraction):
```
python train.py 
```
After training for 250 iterations (~8 hours on a single 2080 Ti), you can find the following model at `model_checkpoint_00250.pt`.

---

To Embedding watermark information in video frames:
```
python test,encoder.py 
```
 (~1 s on a single 2080 Ti), you can find the following stego at `image\steg\0000.png`.

---

If the size of the video frame does not match the size required by the invertible network :
```
python cut.py 
python connet.py 
```
Achieve size consistency
---

To Replace original video frames with watermarked video frames:
```
python CutVideo.py 
```
---
Simulating noise attacks on rendered video frames:
```
file noisey
```
Getting the attacked video frame
---

 To recover video frames:
```
python recover.py
---
Watermark extraction using INN inverse process:
```
python test,decoder.py
```
Watermark extracted successfully
# Acknowledgements

[HNeRV](https://haochen-rye.github.io/HNeRV/#code) models are used to implement MarkINeRV. 
[HiNet](https://github.com/TomTomTommi/HiNet/#code) models are used to implement MarkINeRV.

