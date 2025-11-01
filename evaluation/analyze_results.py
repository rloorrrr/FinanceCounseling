import pandas as pd
import numpy as np
from src.config import CSV_FILE

df = pd.read_csv(CSV_FILE)

print("=" * 70)
print("📊 유사도 평가 결과")
print("=" * 70)

print(f"\n총 질문 수: {len(df)}개")
print(f"평균 유사도: {df['similarity'].mean():.4f}")
print(f"최대 유사도: {df['similarity'].max():.4f}")
print(f"최소 유사도: {df['similarity'].min():.4f}")
print(f"중앙값: {df['similarity'].median():.4f}")
print(f"표준편차: {df['similarity'].std():.4f}")

print("\n" + "=" * 70)
print("📈 등급별 분포")
print("=" * 70)

excellent = len(df[df['similarity'] >= 0.7])
good = len(df[(df['similarity'] >= 0.5) & (df['similarity'] < 0.7)])
fair = len(df[(df['similarity'] >= 0.3) & (df['similarity'] < 0.5)])
poor = len(df[df['similarity'] < 0.3])

print(f"🟢 우수 (0.7 이상): {excellent}개 ({excellent/len(df)*100:.1f}%)")
print(f"🟡 양호 (0.5~0.7): {good}개 ({good/len(df)*100:.1f}%)")
print(f"🟠 보통 (0.3~0.5): {fair}개 ({fair/len(df)*100:.1f}%)")
print(f"🔴 미흡 (0.3 미만): {poor}개 ({poor/len(df)*100:.1f}%)")

print("\n" + "=" * 70)
print("✅ 합격률 분석 (기준: 유사도 0.5 이상)")
print("=" * 70)

pass_threshold = 0.5
pass_count = len(df[df['similarity'] >= pass_threshold])
pass_rate = pass_count / len(df) * 100

print(f"합격: {pass_count}/{len(df)}개")
print(f"합격률: {pass_rate:.1f}%")

if pass_rate >= 80:
    print("평가: 🌟 매우 우수")
elif pass_rate >= 60:
    print("평가: ✅ 양호")
elif pass_rate >= 40:
    print("평가: ⚠️ 보통")
else:
    print("평가: ❌ 개선 필요")

print("\n" + "=" * 70)
print("🏆 최고 점수 질문 TOP 3")
print("=" * 70)

top3 = df.nlargest(3, 'similarity')
for idx, row in top3.iterrows():
    print(f"\n[Q{row['sample_id']}] 유사도: {row['similarity']:.4f}")
    print(f"질문: {row['question']}")
    print(f"실제: {row['real_answer'][:50]}...")
    print(f"모델: {row['model_answer'][:50]}...")

print("\n" + "=" * 70)
print("⚠️ 최저 점수 질문 TOP 3")
print("=" * 70)

bottom3 = df.nsmallest(3, 'similarity')
for idx, row in bottom3.iterrows():
    print(f"\n[Q{row['sample_id']}] 유사도: {row['similarity']:.4f}")
    print(f"질문: {row['question']}")
    print(f"실제: {row['real_answer'][:50]}...")
    print(f"모델: {row['model_answer'][:50]}...")

print("\n" + "=" * 70)
print("📊 유사도 분포")
print("=" * 70)

bins = [0, 0.3, 0.5, 0.7, 1.0]
labels = ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-1.0']
df['range'] = pd.cut(df['similarity'], bins=bins, labels=labels, include_lowest=True)

print(df['range'].value_counts().sort_index())
