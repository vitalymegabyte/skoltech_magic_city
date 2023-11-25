# Команда Magic City, кейс "распознавание инфраструктурных объектов"

Видеодемо: https://drive.google.com/file/d/1W8T7N7y5Jv0exvQhGcMsGL_5W8hy4s5A/view

Для прогона используем команду:

```
python3 predict.py FOLDER/WITH/IMAGES
```

Результат выводится в папку `test_out`

## Загрузка датасетов для предобучения

Запустить код ниже, для загрузки необходимой части датасета SpaceNet
`aws s3 cp s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/Atlanta_nadir7_catid_1030010003D22F00.tar.gz .
aws s3 cp s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/Atlanta_nadir10_catid_1030010003993E00.tar.gz .
aws s3 cp s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/Atlanta_nadir13_catid_1030010002B7D800.tar.gz .
aws s3 cp s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/Atlanta_nadir8_catid_10300100023BC100.tar.gz .
aws s3 cp s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/Atlanta_nadir16_catid_1030010002649200.tar.gz .
aws s3 cp s3://spacenet-dataset/spacenet/SN4_buildings/tarballs/train/Atlanta_nadir10_catid_1030010003CAF100.tar.gz .`
Скачать датасет Alabama Buildings
`https://www.kaggle.com/datasets/meowmeowplus/alabama-buildings-segmentation`
Разархивировать в папки `spacenet` и `alabama` соответственно.