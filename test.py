from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(model_path, sequence, max_length):

    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    # print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

model2_path = 'codethon_data/chat_models/q_and_a'
# What is the Babel fish?"
# the capital city of France
# painted the Mona Lisa
# tallest mountain in the world
max_len = 50

while True:
    user_input = input("Ask me anything (Q) for quit:")
    if user_input == 'q' or user_input == 'Q':
        break
    else:
        user_input = f"[Q] {user_input}"
        max_len = 40
        final_output = generate_text(model2_path, user_input, max_len)
        if '[ANS_END]' in final_output:
            output = final_output.split('[ANS_END]')
            print(output[0])
        else:
            max_len = 50
            final_output = generate_text(model2_path, user_input, max_len)
            output = final_output.split('[ANS_END]')
            print(output[0])

