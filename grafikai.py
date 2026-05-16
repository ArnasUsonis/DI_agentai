import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

modeliai = ['Qwen2.5', 'Qwen3', 'Llama3.1']
spalvos = ['#2ecc71', '#3498db', '#e74c3c']

laikai1 = [45.62, 98.94, 25.59]

plt.figure(figsize=(8, 5))
bars = plt.bar(modeliai, laikai1, color=spalvos, width=0.5)

for bar, laikas in zip(bars, laikai1):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{laikas}s', ha='center', va='bottom', fontweight='bold')

plt.ylabel('Vykdymo laikas (sekundėmis)')
plt.title('Modelių vykdymo laikas pirmajame testiniame klausime')
plt.ylim(0, 120)
plt.tight_layout()
plt.savefig('laikas_1_klausimas.png', dpi=150, bbox_inches='tight')
plt.show()

laikai2 = [8.79, 107.07, 5.14]

plt.figure(figsize=(8, 5))
bars = plt.bar(modeliai, laikai2, color=spalvos, width=0.5)

for bar, laikas in zip(bars, laikai2):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{laikas}s', ha='center', va='bottom', fontweight='bold')

plt.ylabel('Vykdymo laikas (sekundėmis)')
plt.title('Modelių vykdymo laikas antrajame testiniame klausime')
plt.ylim(0, 130)
plt.tight_layout()
plt.savefig('laikas_2_klausimas.png', dpi=150, bbox_inches='tight')
plt.show()