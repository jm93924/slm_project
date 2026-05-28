from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translation():
    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang="kor_Hang"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text = input("한국어 입력 : ")

    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=100
    )

    result = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    print(result)



    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang="eng_Latn"
    )


    text = input("eng input : ")

    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"),
        max_length=100
    )

    result = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    print(result)


if __name__ == "__main__":
    translation()