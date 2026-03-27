import re
import matplotlib.pyplot as plt

def read_metrics(path):
    train = []
    val = []
    epoch_re_tr = re.compile(r"epoch=(\d+) +crossentropy\(train\): ([0-9\.\-]+)")
    epoch_re_va = re.compile(r"epoch=(\d+) +crossentropy\(validation\): ([0-9\.\-]+)")

    with open(path, "r") as f:
        for line in f:
            m = epoch_re_tr.match(line.strip())
            if m:
                train.append(float(m.group(2)))
                continue
            m = epoch_re_va.match(line.strip())
            if m:
                val.append(float(m.group(2)))
                continue


    epochs = list(range(1, len(train) + 1))
    return epochs, train, val

# load both
e_sig, tr_sig, va_sig = read_metrics("sig_metrics.txt")
e_relu, tr_relu, va_relu = read_metrics("relu_metrics.txt")

plt.figure()
plt.plot(e_sig, tr_sig, label="Sigmoid: train")
plt.plot(e_sig, va_sig, label="Sigmoid: validation")
plt.plot(e_relu, tr_relu, label="ReLU: train")
plt.plot(e_relu, va_relu, label="ReLU: validation")

plt.xlabel("Epoch")
plt.ylabel("Average cross-entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig("q4_relu_vs_sigmoid_1hl.png", dpi=200)
plt.show()