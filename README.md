# TextClassifier
A classic model in text classification

基于传统机器学习方法的文本分类，
由于任务本身只用来判断文档中是否含有广告，因此使用词袋模型描述非常合适

使用TF-IDF对文档进行向量化，然后使用回归模型对是否含有广告进行分类
用回归系数解释每一个词汇对于分类结果的贡献率，贡献率大的则最有可能是广告词汇。
可以得出前15个贡献率最高的词汇，和他们的得分，[前2000个词汇和他们的贡献率](./ad_words_2000.txt)：

按摩: 4.079132526461037
枕头: 3.5044737362713723
脖子: 2.454761350461462
颈椎: 1.8747123620076012
电动牙刷: 1.7658402731561411
质量: 1.4879437034413907
电线杆: 1.3356809199568775
评论: 1.2995657229172177
牙齿: 1.1808139535955653
眼睛: 1.178737991892029
冲牙器: 1.152451894782012
直接: 1.07366564700799
设计: 1.0591445007034272
优惠券: 1.0256537368754055
专属: 1.0125196794961997
肩颈: 1.0013138712734246
