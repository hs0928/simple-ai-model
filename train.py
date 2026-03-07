model.train()
for iter in range(5000):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"단계 {iter}: 오차 {loss.item():.4f}")
