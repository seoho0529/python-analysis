import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from functools import reduce
plt.rc('font',family='malgun gothic')
import scipy.stats as stats

print('연도CPI')
cpi = pd.read_csv("../testdata/CPI2.csv", index_col='주요품목별(1)')
# print(cpi)
cpi_mean = pd.DataFrame(cpi.mean(),)
cpi_mean.columns=['CPI']
cpi_mean.index.name = '연도'
cpi_mean.reset_index(inplace=True)
cpi_mean = cpi_mean.set_index('연도')
# print(cpi_mean.index)
# print(cpi_mean.columns)
print(cpi_mean, type(cpi_mean)) # 연도 CPI

# print(cpi_mean, type(cpi_mean))
# print(cpi_mean)
# plt.plot(cpi_mean)
# plt.show()

unempl = pd.read_csv('../testdata/unemployment.csv', index_col='연도')
# print(type(unempl))
# imsi = unempl.T
# imsi.columns = imsi.iloc[0]
# imsi=imsi[1:]
# print(imsi)
# unempl = imsi.T
# unempl.set_index('연도')
# print(unempl.index)
# print(unempl.columns)
print(unempl, type(unempl)) # 연도 실업률

print('스트레스')
stress = pd.read_csv('../testdata/stress2.csv', index_col=None)
stress = stress.T
stress = stress.iloc[1:,]
stress.columns=['스트레스 인지율']
stress.index.name = '연도'
stress.reset_index(inplace=True)
stress = stress.set_index('연도')
print(stress)  # 연도 스트레스인지율

# plt.plot(stress)
# plt.show()
print('인구 수')
popul = pd.read_csv('../testdata/population.csv')
print(popul)
imsi = popul.iloc[1, 3:]
print(imsi)

popul_tot = pd.DataFrame(imsi)
popul_tot.columns=['인구수']
popul_tot.index.name = '연도'
popul_tot['연도']=popul_tot.index
popul_tot.reset_index(drop=True, inplace=True)
popul_tot=popul_tot[['연도','인구수']]
popul_tot = popul_tot.set_index('연도')
print(popul_tot)

print('범죄 발생건수')
crime = pd.read_csv('../testdata/crime.csv',)
print(crime)
crime = crime.T
crime.reset_index(inplace=True)
crime.columns = crime.iloc[0]
crime = crime[1:]  # 첫 번째 행을 열 이름으로 사용했으므로, 이후의 행은 데이터로 간주하고 첫 번째 행을 제거합니다.
crime.index.name = None  # 행에 대한 이름을 제거합니다.
crime = crime.set_index('연도')
crime.columns = ['발생건수']
print(crime)

# unempl['연도'] = unempl['연도'].astype(object)
cpi_mean.index = cpi_mean.index.astype(int)
stress.index = stress.index.astype(int)
popul_tot.index = popul_tot.index.astype(int)
popul_tot['인구수'] = popul_tot['인구수'].astype(int)
crime.index = crime.index.astype(int)
crime['발생건수'] = crime['발생건수'].astype(int)
print(stress['스트레스 인지율'].dtype)
stress['스트레스 인지율']=stress['스트레스 인지율'].astype('float64')
print(stress['스트레스 인지율'].dtype)
# print(cpi_mean['연도'].dtypes, unempl['연도'].dtypes)
# cpi_unemple = pd.merge(cpi_mean, unempl, on='연도')
# print(cpi_unemple)



print('데이터 프레임 합치기-crime_anal')
df = [cpi_mean, unempl, stress, popul_tot, crime]
print(df)
crime_analysis = reduce(lambda left, right: pd.merge(left, right, on='연도', how='inner'), df)
print(crime_analysis)
# crime_anal=crime_analysis.set_index('연도') # 컬럼에 있던 연도를 인덱스로 바꿈
# print(crime_anal)

# (CPI, 실업률, 스트레스 인지율, 인구수, 발생건수) 산점도 그리기
import matplotlib.pylab as pylab  
scatter_matrix(crime_analysis, alpha=0.8, diagonal='hist')
#plt.show()

#  상관계수 구하기
print(crime_analysis.corr())  
#                CPI       실업률  스트레스 인지율       인구수        발생건수
# CPI       1.000000 -0.139293 -0.851870 -0.949905 -0.890489
# 실업률      -0.139293  1.000000  0.109529  0.076489 -0.161952
# 스트레스 인지율 -0.851870  0.109529  1.000000  0.889252  0.865526
# 인구수      -0.949905  0.076489  0.889252  1.000000  0.903435
# 건        -0.890489 -0.161952  0.865526  0.903435  1.000000
# 실업률은 발생건수와 상관관계가 거의 없다고 알 수 있다.

# 히트맵
import seaborn as sns
sns.pairplot(crime_analysis, kind='reg')
#plt.show()

# 피어슨 상관계수로 각 독립변수와 종속변수의 상관계수 구하기
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['CPI'])) # PearsonRResult(statistic=-0.8904886218046394, pvalue=8.58853600470416e-06)
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['실업률'])) # PearsonRResult(statistic=-0.16195175313674026, pvalue=0.5641760129646295)
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['스트레스 인지율'])) # PearsonRResult(statistic=0.8655261708093139, pvalue=3.065014812312715e-05)
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['인구수']))  # PearsonRResult(statistic=0.903434689304182, pvalue=3.915084633601483e-06)
# 이때, '실업률' 변수를 제외한 나머지 3개의 변수는 모두 0.05보다 작으므로 종속변수인 범죄건수에 유의한 영향을 미칠 것으로 간주할 수 있다.

# bartlett을 사용한 등분산 검정
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['CPI']))
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['실업률']))
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['스트레스 인지율']))
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['인구수']))
# 4개 변수 모두 pvalue가 0.05보다 작으므로 각 변수들이 통계적으로 유의하다.는 것을 알 수 있다.

# 독립성 검정 shapiro 사용하기
from scipy.stats import shapiro
shapiro_results = crime_analysis.apply(lambda x: shapiro(x))
print(shapiro_results)
#       CPI      실업률   스트레스 인지율  인구수     발생건수
# 0  0.975572  0.915725  0.916385  0.923464  0.965646
# 1  0.930298  0.165725  0.169626  0.217494  0.789177
# 인덱스 0은 검정통계량, 1번째는 pvalue를 나타내는데 4변수들의 pvalue 모두 0.05보다 크므로 4변수 모두 정규성을 만족한다고 볼 수 있다.

# 위 결과를 종합적으로 보았을 때 실업률을 제외하고 3가지 변수들에 대해 ANOVA분산분석을 실시하겠다.

from scipy import stats
# 등분산 검정
print(stats.bartlett(crime_analysis['CPI'],crime_analysis['실업률'], crime_analysis['스트레스 인지율'], crime_analysis['인구수']),stats.fligner(crime_analysis['CPI'],crime_analysis['실업률'], crime_analysis['스트레스 인지율'], crime_analysis['인구수']) ,stats.levene(crime_analysis['CPI'],crime_analysis['실업률'], crime_analysis['스트레스 인지율'], crime_analysis['인구수']), sep="\n")
# BartlettResult(statistic=901.4731487208453, pvalue=4.241215932978223e-195)
# FlignerResult(statistic=39.57852798144273, pvalue=1.3088388445622003e-08)
# LeveneResult(statistic=26.629502809814117, pvalue=7.79363288050101e-11)



"""
# 다중선형회귀분석 모델 생성 하기위해 독립,종속 변수 선택 및 모델 생성
from sklearn.linear_model import LinearRegression

X=crime_analysis[['CPI','실업률','스트레스 인지율','인구수']]
Y=crime_analysis['발생건수']

X.columns=[['cpi','unemp','stress','pop']]
Y.columns=['crimecount']
print(X)
print(Y)

model = LinearRegression()
model.fit(X,Y)

# 다중회귀분석이기 때문에 변수들 간 다중공선성 확인
X=crime_analysis[['CPI','실업률','스트레스 인지율', '인구수']]
Y=crime_analysis['범죄건수']

print(stats.f_oneway(X))
"""

"""
# 예측 모델 생성
y_pred = model.predict(X)

# 가중치와 y절편 출력
print('가중치 a : ',model.coef_)
print('y절편 : ',model.intercept_)

# 결정계수
r_value = model.score(X,Y)
print("결정계수 : ",r_value)

# 다중회귀식은
# 발생 건수=−2,828.5×CPI−30,174×실업률+6,379.7×스트레스 인지율+0.022×인구수+328,526.86 로 구할 수 있다.

# seaborn으로 나타내기
import seaborn as sns
ax1 = sns.displot(Y, label='Y실제 값', kde=True)
ax2 = sns.displot(y_pred, label='Y예측 값', kde=True)
plt.show()
"""