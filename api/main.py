import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "tech4_secret_key"

model = joblib.load("models/model.joblib")
expected_columns = model.named_steps["preprocessor"].feature_names_in_


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    def to_float(valor):
        """Converte número com vírgula ou ponto"""
        if valor is None or valor == "":
            return 0.0
        return float(valor.replace(",", "."))

    def to_int(valor):
        if valor is None or valor == "":
            return 0
        return int(valor)

    # 🔹 Dados do formulário
    idade = to_float(request.form.get("idade"))
    inde22 = to_float(request.form.get("inde22"))
    inde23 = to_float(request.form.get("inde23"))
    inde24 = to_float(request.form.get("inde24"))
    ativo = to_int(request.form.get("ativo"))

    # 🔹 Features criadas na API (igual no treino)
    media_inde = (inde22 + inde23 + inde24) / 3
    qtde_abaixo_6 = sum([inde22 < 6, inde23 < 6, inde24 < 6])

    # 🔹 Criar DataFrame com TODAS as colunas que o modelo espera
    input_data = {col: [0] for col in expected_columns}

    # 🔹 Preencher apenas as que realmente usamos
    if "Idade" in input_data:
        input_data["Idade"] = [idade]

    if "INDE 2022" in input_data:
        input_data["INDE 2022"] = [inde22]

    if "INDE 2023" in input_data:
        input_data["INDE 2023"] = [inde23]

    if "INDE 2024" in input_data:
        input_data["INDE 2024"] = [inde24]

    if "MEDIA_INDE" in input_data:
        input_data["MEDIA_INDE"] = [media_inde]

    if "QTDE_INDE_BAIXO_6" in input_data:
        input_data["QTDE_INDE_BAIXO_6"] = [qtde_abaixo_6]

    if "Ativo/ Inativo" in input_data:
        input_data["Ativo/ Inativo"] = [ativo]

    input_df = pd.DataFrame(input_data)

    # 🔹 Previsão
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    risco = "ALTO RISCO" if prediction == 1 else "BAIXO RISCO"

    # 🔹 Histórico
    if "historico" not in session:
        session["historico"] = []

    session["historico"].append({
        "idade": idade,
        "media_inde": round(media_inde, 2),
        "qtde_baixo6": qtde_abaixo_6,
        "ativo": "Sim" if ativo == 1 else "Não",
        "risco": risco,
        "prob": round(probability, 1)
    })

    session.modified = True

    return redirect(url_for("resultado"))



@app.route("/resultado")
def resultado():
    historico = session.get("historico", [])
    return render_template("resultado.html", historico=historico)

@app.route("/limpar")
def limpar():
    session.pop("historico", None)
    return redirect(url_for("resultado"))



@app.route("/nova")
def nova():
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
