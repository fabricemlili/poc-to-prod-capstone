# POC TO PROD

To run the application flask, you need to run  ``` python3 -m predict.predict.app ```  in your terminal. All the project is unit tested.

---

## Preprocessing

The loading and preprocessing of the dataset.

---

## Train

train-conf.yml hold all the parameters and you can change them. To lauch a train use ``` python3 -m train.train.run train/data/training-data/stackoverflow_posts.csv train/conf/train-conf.yml train/data/artefacts ``` and don't forget to change the path of the artefacts in the run in predict/predict.

---

## Predict

The link to access to the app in local is http://127.0.0.1:5000/. Enter a texte in the input block and submit, then watch the 5 most probable predictions. 

---

## Improvement

Some improvements need to be done. Firstly, the model is very bad and has an accuracy of 10%. Secondly, the output layer is too big because the categorisation is badly pre-process.

---