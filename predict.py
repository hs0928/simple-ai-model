def ask(question):
    prompt = f"q: {question} a:"
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for  in range(50):
        logits,  = model(idx[:, -block_size:])
        #idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        probs = F.softmax(logits[:, -1, :] / 0.6, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    output = decode(idx[0].tolist())
    return output.split("a:")[1].split("q:")[0].strip()

print(ask("한국 수도 어디야"))
