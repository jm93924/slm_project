from transformers import pipeline

MODEL_NAME = "beomi/KcELECTRA-base-v2022"

def sentiment_analysis():
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

    while True:
        text = input("\n문장 입력 (종료: exit/quit/q) : ")

        if text.lower() in ["exit", "quit", "q"]:
            break

        result = classifier(text)[0]

        label = result["label"]
        confidence = result["score"]

        # 점수 보정
        if label == "LABEL_1":
            score = round(confidence * 100)
            sentiment = "긍정 😀"
        else:
            score = round((1 - confidence) * 100)
            sentiment = "부정 😡"

        print("\n===== 분석 결과 =====")
        print("입력 문장 :", text)
        print("예측 라벨 :", label)
        print("감성 분석 :", sentiment)
        print("신뢰도 점수 :", f"{score}점")

        print()
        print(result)

if __name__ == "__main__":
    sentiment_analysis()