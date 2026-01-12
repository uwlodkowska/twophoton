import pandas as pd, matplotlib.pyplot as plt
phases = pd.DataFrame({
    "phase": ["Implantacja","Habituacja","trening","Washout","Probe"],
    "days":  [4, 15, 7, 2],
    "note":  ["Handling","Task A","No task","Final test"]
})
phases["start_day"] = phases["days"].shift(fill_value=0).cumsum() - phases["days"]
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.barh(range(len(phases)), phases["days"], left=phases["start_day"])
ax.set_yticks(range(len(phases)), phases["phase"])
ax.set_xlabel("Day"); ax.set_title("Experiment timeline (relative)")
for yi, s, w, note in zip(range(len(phases)), phases["start_day"], phases["days"], phases["note"]):
    ax.text(s + 0.01*w, yi, note, va="center", ha="left", fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 2.8))
# each tuple is (start, length); second arg is (ymin, height)
ax.broken_barh([(-6,2)], (3.5, 0.8))
ax.broken_barh([(-4,4)], (2.5, 0.8))
ax.broken_barh([(0,2), (2,2), (4,10)], (1.5, 0.8), facecolors=["#8dc2e4", "#68b0de", "tab:blue"])
ax.scatter([15], [0.9], c="orange", s=35, marker="D")
ax.scatter([16], [0.9], c="green", s=40, marker="^")
ax.scatter([17], [0.9], c="green", s=40, marker="^")

ax.scatter([20,20.5,21], [0.9]*3, c="black", s=2)

ax.scatter([24], [0.9], c="orange", s=35, marker="D")
ax.scatter([25], [0.9], c="green", s=40, marker="^")
ax.scatter([26], [0.9], c="green", s=40, marker="^")
ax.scatter([27], [0.9], c="tab:blue", s=40, marker="o")
ax.scatter([-7], [4.9], c="red", s=40)
ax.set_ylim(-0.5, 6)
ax.set_xlim(-8, 28)
ax.set_yticks([4.9, 3.9, 2.9, 1.9, 0.9], ["Implantacja", "Rekonwalescencja", "Habituacja", "Trening", "Test"], fontsize=11, rotation=15)
ax.set_xticks([-5.2, -2,7, 16,20.5, 25.5], ["~7 dni", "14 dni", "13-20 dni", "3 dni","4 tygodnie", "4 dni"], fontsize=11)
ax.set_title("Schemat przebiegu eksperymentu 1", pad = 14, fontsize=16)
ax.text(1, 1.9, "16", ha="center", va="center", fontsize=9, color="black")
ax.text(3, 1.9, "26", ha="center", va="center", fontsize=9, color="black")
ax.text(9, 1.9, "50", ha="center", va="center", fontsize=9, color="white")

ax.text(15, 0, "A", ha="center", va="center", fontsize=9, color="black")
ax.text(16, 0, "B", ha="center", va="center", fontsize=9, color="black")
ax.text(17, 0, "B", ha="center", va="center", fontsize=9, color="black")
ax.text(24, 0, "A", ha="center", va="center", fontsize=9, color="black")
ax.text(25, 0, "B", ha="center", va="center", fontsize=9, color="black")
ax.text(26, 0, "B", ha="center", va="center", fontsize=9, color="black")
ax.text(27, 0, "T", ha="center", va="center", fontsize=9, color="black")
ax.text(25.5, 1.5, "Retrieval", ha="center", va="center", fontsize=10, color="black")

plt.tight_layout()
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 2.8))
# each tuple is (start, length); second arg is (ymin, height)

ax.broken_barh([(-6,6)], (2.5, 0.8))
ax.broken_barh([(0,2), (2,2), (4,10)], (1.5, 0.8), facecolors=["#8dc2e4", "#68b0de", "tab:blue"])
ax.scatter([15], [0.8], c="orange", s=95, marker="*")
ax.scatter([16], [0.8], c="green", s=40, marker="o")

ax.set_ylim(-0.75, 4.6); ax.set_xlim(-7, 18)
ax.set_yticks([2.9, 1.9, 0.9], ["Habituacja", "Trening", "Test"], fontsize=12)
ax.set_xticks([-3.5, 7, 15.5], ["14 dni", "6-16 dni", "2 dni"], fontsize=12)

ax.text(1, 1.9, "16", ha="center", va="center", fontsize=9, color="black")
ax.text(3, 1.9, "32", ha="center", va="center", fontsize=9, color="black")
ax.text(9, 1.9, "50", ha="center", va="center", fontsize=9, color="white")

ax.set_title("Schemat przebiegu eksperymentu 3", pad = 14, fontsize=16)
ax.axvline(x=16.5, ymin=-0.75, ymax=4.6, color="red", linestyle=":")
ax.text(17, 2, "Terminacja: test + 90'", ha="center", va="center", fontsize=12, color="red", rotation="vertical")
# ax.text(3, 1.9, "26", ha="center", va="center", fontsize=9, color="black")
# ax.text(9, 1.9, "50", ha="center", va="center", fontsize=9, color="white")

# ax.text(15, 0, "A", ha="center", va="center", fontsize=9, color="black")
# ax.text(16, 0, "B", ha="center", va="center", fontsize=9, color="black")
# ax.text(17, 0, "B", ha="center", va="center", fontsize=9, color="black")

plt.tight_layout()

#%%
