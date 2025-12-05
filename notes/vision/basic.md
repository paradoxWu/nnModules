* 为什么resnet中会选择小核的卷积而不是大核
  * 以kernel_size3与7的两个conv2d来说，三层的conv2d(3)感受野是与conv2d(7)一致
  * conv2d(3) 堆叠一次后 感受野等效conv2d（5），堆叠两次等效conv2d（7）
  * 在同感受野的前提下，对于rgb图片（channel=3），参数量更小，结构更深（非线性操作空间更大）
  * 参数量：
    * 3个conv2d(3):  c=3   / 3* (3 * 3  * 3) = 81
    * 1个conv2d(7): c =3 /  7* 7* 3 = 147
* renet的核心
  * skip connection，残差
  * 深层会导致退化，根源是梯度消失
  * 在梯度消失时，至少能够复制输入
  * 输入x，经过F，得F(x), 输出H(x) = F(x)+x,若梯度消失，F(x) = 0, 那么输出H(x) = x，至少不会变差
  * 手写basicblock bottleneck
* 卷积计算
  * input_C,output_C,kernel_size, stride,padding 决定了output的shape
  * input shape为 （in_c，x，x）
  * output shape 为 （out_c，(x-kernel_size)/stride+padding),(x-kernel_size)/stride+padding)
* padding补0为什么有效？
  1. **零值可被学习**
     CNN 的权重 **自己会学会「忽略 0」或「利用 0」** ——只要该位置对任务有用，梯度就会让权重放大它；若没用，权重趋近于 0 即可屏蔽。
  2. **边界信息本身就是特征**
     在很多任务里， **“物体是否触边”就是重要线索** （检测、分割、轨迹边界等），0-padding 恰好把这些「边缘模式」显式暴露给网络，反而帮助判别。
  3. **零值对称，不引入偏置**
     与某些「常量填充」不同，0 是 **中性值** ，均值=0，不会把整个 feature map 的统计量拉偏；BatchNorm/LayerNorm 一步就能重新中心化。
  4. **经验结果：Zero-padding 足够好**
     ImageNet/COCO/Argoverse 等大量 SOTA 模型 **清一色用 zero-padding** ，精度并未劣于 reflect/mirror 等更复杂填充—— **数据+正则足够时，网络会自动学出边界处理策略** 。
  5. **复杂填充只是「锦上添花」**
     Reflect/Replicate 等填充 **仅在输入数据本身有强连续性时** （医学图像、自然风景）略优于 zero；对于**离散、人工、有遮挡**的输入（轨迹图、激光雷达 BEV），zero-padding 往往更鲁棒。
* Bottleneck =「 **先降维 → 3×3 小卷积 → 再升维** 」的残差块， **用更少的参数获得更深的网络** ，是 ResNet-50/101/152 的标配。核心目标： **在增加深度的同时，不增加计算量** 。
