- 这里写了conv的fp32方法
- cutlass也可以用fp32，不过layout居然只能用NCHW,而且速度也比cuDNNDNN 慢啊


# 3x3s1， 1xx32xx112xx112，32

- 也就是resnet50中的第二个卷积。
- cutlass目前来看，似乎是仅仅支持NHWC！

| T4         | fp32     |
| ---------- | -------- |
| cutlass    | 1.777536 |
| cuDNN/NCHW | 1.011264 |
| cuDNN/NHWC | 2.296032 |







