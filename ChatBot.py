from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import BaseLLM
from langchain.chains import ConversationChain

class MyLLM(BaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _call(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _identifying_params(self):
        return {"model_name": self.model.config._name_or_path}

# 替换为你微调后的模型路径
model_path = "../finetuned_model"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 创建MyLLM实例
llm = MyLLM(model, tokenizer)

# 创建对话链
conversation_chain = ConversationChain(llm=llm)

# 聊天函数
def chat_with_bot():
    print("你可以开始与机器人聊天了！输入 'exit' 结束对话。\n")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            print("对话结束。")
            break
        response = conversation_chain.predict(input=user_input)
        print(f"机器人: {response}")

if __name__ == "__main__":
    chat_with_bot()
