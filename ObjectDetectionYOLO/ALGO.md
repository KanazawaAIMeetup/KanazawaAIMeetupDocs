IoUはintersection over unionの略
<img src="https://meideru.com/wp-content/uploads/2017/05/iou_value_1.png">
予測された複数のバウンディングボックスのうち、重なっている部分が大きい場合は一つに絞る。
逆に重なっている部分が小さい、もしくは全く無いバウンディングボックスの場合は、消去せずに残す。
- Non-Maximum Suppression(https://meideru.com/archives/3538)

<img src="region.png">
![image](region.png)
<img src=”region.png” />
