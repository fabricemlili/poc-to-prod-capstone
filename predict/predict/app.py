from flask import Flask, request, render_template
import os

# Create a Flask app
app = Flask(__name__)

input_txt = None

# Define a route for the default page
@app.route('/', methods=['GET', 'POST'])
def input_txt():
    # dataset_path = "poc-to-prod-capstone/train/data/training-data/stackoverflow_posts.csv"
    # train_conf = {
    #             "batch_size": 2,
    #             "epochs": 1,
    #             "dense_dim": 64,
    #             "min_samples_per_label": 1,
    #             "verbose": 1
    #         }
    # model_path = "poc-to-prod-capstone/train/data/artefacts"
    # add_timestamp = True
    # run_train.train(dataset_path, train_conf, model_path, add_timestamp)

    return """    Write a stackoverflow topic here: <br><br>
     <form action="/prediction" method="post">
      <input type="text" name="sentence">
      <input type="submit" value="Submit">
    </form>"""


@app.route('/prediction', methods=['GET', 'POST'])
def display_prediction():

    input_txt = request.form['sentence']
    
    output = os.popen(f'python3 -m predict.predict.run ./train/data/artefacts/2023-01-06-17-55-58 {input_txt}').read()
    return f"""{output} <br><br><br><br> <a href="/">Do it again</a>"""


# Run the app
if __name__ == '__main__':
    app.run()