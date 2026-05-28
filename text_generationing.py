import textwrap
import transformers
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import time

def text_generation():
    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>",       # 문장의 시작을 의미하는 토큰
        eos_token="</s>",       # 문장의 끝을 의미하는 토큰
        unk_token="<unk>",      # 모르는 토큰을 의미하는 토큰
        pad_token="<pad>",      # 길이를 맞추기 위해 채우는 토큰
        mask_token="<mask>",     # 가려진 토큰을 의미(BERT류에서 많이 사용)
        )

    pipe = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    prompt = input("input : ")

    result = pipe(prompt,
                  #max_length=50,      # 입력+출력 전체 길이
                  max_new_tokens=100, # 새로 생성할 출력 길이
                  do_sample=True,    # 창의적이지만 가끔 헛소리
                  #do_sample=False,   # 안정적이고 보수적. default값.
                  top_k=50,           # 확률이 높은 50개 단어만 고려
                  top_p=0.95,         # 1등 단어부터 누적하여 95%만 고려
                  repetition_penalty=1.2,   # 단어 반복사용 규제
                  temperature=0.7,          # 창의성 강도 조절
                  eos_token_id=tokenizer.eos_token_id,      # EOS 토큰이 나오면 생성을 멈춰라
                  pad_token_id=pipe.tokenizer.eos_token_id  # pad 토큰을 EOS토큰 대신 써라(경고 방지)=
                  )
    
    generated_text = result[0]["generated_text"]

    print("Input text :", prompt)
    print("Output text : ")
    print(textwrap.fill(generated_text, width=60))

if __name__ == "__main__":
    text_generation()