import json
import matplotlib.pyplot as plt

with open('./output/Pretrain/log.txt', 'r') as file:
    data = file.read()
parsed_data = json.loads('[' + data.replace('}\n{', '},{') + ']')

epochs = [item['epoch'] for item in parsed_data]
loss_ita = [float(item['train_loss_ita']) for item in parsed_data]
loss_itm = [float(item['train_loss_itm']) for item in parsed_data]
loss_lm = [float(item['train_loss_lm']) for item in parsed_data]

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_ita, label='ITA Loss')
plt.plot(epochs, loss_itm, label='ITM Loss')
plt.plot(epochs, loss_lm, label='LM Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss graph per epoch')
plt.legend()
plt.grid(True)
plt.show()