from flask import Flask, render_template, request
import g4f
import joblib
import pandas as p

app = Flask(__name__)

model = joblib.load("crop_yield_predictor.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def calc():
    crop = request.form["crop"]
    temp = request.form["temperature"]
    rain = request.form["rainfall"]
    soil = request.form["soil"]
    mois = request.form["moisture"]
    pest = request.form["pest"]

    pred = model.predict([[float(rain), float(temp)]])

    response = ""
    for provider in [
        g4f.Provider.ChatgptAi,
        g4f.Provider.FreeGpt,
        g4f.Provider.FakeGpt,
        g4f.Provider.GPTalk,
        g4f.Provider.GptForLove,
    ]:
        try:
            response = g4f.ChatCompletion.create(
                messages=[
                    {
                        "role": "assistant",
                        "content": "Hello! I'm a crop yield predicting chatbot who provides less detailed answer by stressing on the information provided! I will only give answers in less than 60 words. I do not ask questions to users. I do not start my sentences with openers and don't end them with closers. I do not mention if other factors might affect. I do not mention if it's difficult for me to answer.",
                    },
                    {
                        "role": "user",
                        "content": f"I have sown {crop} in a temperature of {temp} with a rainfall of {rain} in a soil of type {soil} with a moisture of {mois} and have used {pest} pesticides/fertilisers. Can you give me a summary of the crop yield with this information and not ask for more information?",
                    },
                ],
                provider=provider,
                model=g4f.models.gpt_35_turbo,
            )
        except:
            continue
        if len(response) != 0:
            break

    return render_template("index.html", summary=response, yyy=pred[0])


if __name__ == "__main__":
    app.run(debug=True)
