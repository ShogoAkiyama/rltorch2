# 使用するパッケージの宣言
import numpy as np
import matplotlib.pyplot as plt

# 初期位置での迷路の様子

# 図を描く大きさと、図の変数名を宣言
fig = plt.figure(figsize=(4, 4))
ax = plt.gca()

# 状態を示す文字S0～S8を描く
plt.text(1, 1.3, 'START', ha='center')
plt.text(1, 10.3, 'GOAL', ha='center')

# 描画範囲の設定と目盛りを消す設定
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')

# Start: green, Goal: blue, Hole: red
ax.plot([1], [1], marker="o", color='g', markersize=40, alpha=0.8)
ax.plot([1], [10], marker="o", color='b', markersize=40, alpha=0.8)
ax.plot([1], [4], marker="x", color='r', markersize=40, markeredgewidth=10)
ax.plot([1], [7], marker="x", color='r', markersize=40, markeredgewidth=10)

plt.show()
