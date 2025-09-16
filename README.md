<img width="543" height="465" alt="image" src="https://github.com/user-attachments/assets/12a95df9-3519-42a3-aae4-a23600529374" />#阿君的蜜汁小工坊#

Sam小工具更新

#1 加入 目标检测/分割 数据保存功能 - 包含 （标签.txt，类别.txt)
<img width="283" height="58" alt="image" src="https://github.com/user-attachments/assets/aaed2e4a-ae61-42c0-a6a6-d5fd0cf2df7d" />
#2 加入 工作区域缩放功能（ctrl+滚轮）/ 加入 可仅分割当前缩放工作区域的内容的功能（小目标分割十分适用）
#3 加入 类别菜单功能（可快速对分割目标进行定义）
<img width="1045" height="869" alt="image" src="https://github.com/user-attachments/assets/efba8f58-2143-47d0-926b-3ed1c51aa0c9" />
#4 加入 锚点/掩码 便捷删除功能，目前鼠标指针移动到锚点时（未选定绿色，选定为红色），选定状态时右键即可删除；掩码部分选定时掩码会变色。
<img width="748" height="316" alt="image" src="https://github.com/user-attachments/assets/f82bb2b0-be37-4a85-9bd3-ddcee8c86fc4" />
#5 加入 标签载入功能（用于检验标注标签是否有问题）
#6 加入 各类操作快捷键
#7 优化 更多可视化效果（最小内接矩阵框 / 掩码选定 / 锚点选定）
#8 优化 部分按钮及菜单使用逻辑（加载，保存，重置）

下一次更新预计加入
①：掩码坐标与内接矩阵坐标的自定义调整功能（目前是全部由seg结果决定）
②：负样本锚点（ROI的反义区域）
③：VLM数据集保存方式
Json
{
  Box：(xy,xy)
  Description：目标是……
}



适用于批量图像裁剪、分割-生成数据集

①Crop哪儿都可以运行，可生成裁剪后的图片以及坐标信息。

②Sam请放在Sam2主目录里运行，或者自行补上sam的yaml文件，需要自行下载一下Sam预训练模型；已经写好相关载入逻辑，目前有的预训练模型都可以使用，分割的结果包含图片掩码及其坐标（后续会弄一下生成目标最小矩形框用以检测任务/yolo格式的txt输出）

后续用到什么就会开发相关的工具，以便高效工作，有什么改进意见或者想法的可以提给我~
