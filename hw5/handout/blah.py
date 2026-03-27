import re
import matplotlib.pyplot as plt

def read_metrics(path):
    train = []
    val = []
    tr_re = re.compile(r"epoch=(\d+) crossentropy\(train\): ([0-9\.\+\-]+)")
    va_re = re.compile(r"epoch=(\d+) crossentropy\(validation\): ([0-9\.\+\-]+)")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            m = tr_re.match(line)
            if m:
                train.append(float(m.group(2)))
                continue
            m = va_re.match(line)
            if m:
                val.append(float(m.group(2)))

    epochs = list(range(1, len(train) + 1))
    return epochs, train, val

eS, trS, vaS = read_metrics("metrics_2hl_sigmoid.txt")
eR, trR, vaR = read_metrics("metrics_2hl_relu.txt")

plt.figure()
plt.plot(eS, trS, label="Sigmoid: train")
plt.plot(eS, vaS, label="Sigmoid: validation")
plt.plot(eR, trR, label="ReLU: train")
plt.plot(eR, vaR, label="ReLU: validation")

plt.xlabel("Epoch")
plt.ylabel("Average cross-entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("q4_relu_vs_sigmoid_2hl.png", dpi=200)