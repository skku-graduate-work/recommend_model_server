# Flask 앱에서 get_similar_foods 함수 수정

from flask import Flask, request, jsonify
from flask_cors import CORS  # 추가된 부분
from gensim.models import Word2Vec
import pandas as pd
import re
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 추가된 부분

# 모델 로드
model_path = "./food2vec.model"
food2vec_model = Word2Vec.load(model_path)

# 데이터프레임 로드
df_path = "./result.txt"
with open(df_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

pattern = re.compile(r"\((\d+),\s([^\)]+)\)")
data = [
    (int(match.group(1)), match.group(2))
    for line in lines
    for match in re.finditer(pattern, line)
]
df = pd.DataFrame(data, columns=["id", "foodname"])
df = df.sort_values(by="id")
df = df.drop_duplicates(subset="id", keep="first")


@app.route("/get_similar_foods", methods=["POST"])
def get_similar_foods():
    try:
        # 클라이언트로부터 입력 음식 리스트 받기
        input_food_list = request.json["input_food_list"]

        # 음식 벡터 평균 계산
        vectors = [
            food2vec_model.wv[word]
            for word in input_food_list
            if word in food2vec_model.wv
        ]

        # 입력 음식들과 유사한 10개의 음식 찾기 (입력 값은 제외)
        if vectors:
            average_vector = sum(vectors) / len(vectors)
            similar_food = food2vec_model.wv.similar_by_vector(average_vector, topn=10)

            # 결과 생성
            result = []
            for food, similarity in similar_food:
                # id 찾아내기
                id_value = df[df["foodname"] == food]["id"].values[0]
                if food not in input_food_list:
                    result.append(
                        {
                            "id": int(id_value),
                            "foodname": food,
                            "similarity": float(similarity),
                        }
                    )

            return jsonify({"success": True, "result": result})

        else:
            return jsonify({"success": False, "message": "입력된 음식이 모델에 없습니다."})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
