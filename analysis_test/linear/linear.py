import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from functools import reduce
from sklearn.preprocessing import StandardScaler
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

outlier_index = cpi_mean[cpi_mean['CPI'] > cpi_mean['CPI'].mean() + 1.5 * cpi_mean['CPI'].std()].index
cpi_mean.loc[outlier_index, 'CPI'] = cpi_mean['CPI'].mean()

# print(cpi_mean, type(cpi_mean))
# print(cpi_mean)
plt.boxplot(cpi_mean)
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

lower_threshold = 3.8 
outlier_index2 = unempl[unempl['실업률'] < lower_threshold].index
unempl.loc[outlier_index2, '실업률'] = unempl['실업률'].mean()
print('바뀐 unempl', unempl)


# 결과 확인
print(unempl)
print('스트레스')
stress = pd.read_csv('../testdata/stress2.csv', index_col=None)
stress = stress.T
stress = stress.iloc[1:,]
stress.columns=['스트레스_인지율']
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
print(stress['스트레스_인지율'].dtype)
stress['스트레스_인지율']=stress['스트레스_인지율'].astype('float64')
print(stress['스트레스_인지율'].dtype)
# print(cpi_mean['연도'].dtypes, unempl['연도'].dtypes)
# cpi_unemple = pd.merge(cpi_mean, unempl, on='연도')
# print(cpi_unemple)



print('데이터 프레임 합치기-crime_anal')
df = [cpi_mean, unempl, stress, popul_tot, crime]
# print(df)
crime_analysis = reduce(lambda left, right: pd.merge(left, right, on='연도', how='inner'), df)
# print(crime_analysis)
# crime_anal=crime_analysis.set_index('연도') # 컬럼에 있던 연도를 인덱스로 바꿈
# print(crime_anal)

# (CPI, 실업률, 스트레스_인지율, 인구수, 발생건수) 산점도 그리기
import matplotlib.pylab as pylab  
scatter_matrix(crime_analysis, alpha=0.8, diagonal='hist')
#plt.show()

#  상관계수 구하기
print(crime_analysis.corr())  
#                CPI       실업률  스트레스_인지율       인구수      발생건수
# CPI       1.000000  0.434024 -0.663881 -0.816641 -0.836556
# 실업률       0.434024  1.000000 -0.342577 -0.347722 -0.553590
# 스트레스_인지율 -0.663881 -0.342577  1.000000  0.889252  0.865526
# 인구수      -0.816641 -0.347722  0.889252  1.000000  0.903435
# 발생건수     -0.836556 -0.553590  0.865526  0.903435  1.000000
# 실업률은 발생건수와 상관관계가 거의 없다고 알 수 있다.

# plt.boxplot(unempl)
# plt.show()
# plt.boxplot(stress)
# plt.show()
# plt.boxplot(popul_tot)
# plt.show()
# plt.boxplot(crime)
# plt.show()

# 정규화
scaler = StandardScaler()

cpi_mean[['CPI']] = scaler.fit_transform(cpi_mean[['CPI']])
unempl[['실업률']] = scaler.fit_transform(unempl[['실업률']])
stress[['스트레스_인지율']] = scaler.fit_transform(stress[['스트레스_인지율']])
popul_tot[['인구수']] = scaler.fit_transform(popul_tot[['인구수']])
crime[['발생건수']] = scaler.fit_transform(crime[['발생건수']])

# print(cpi_mean)
# print(unempl)
# print(stress)
# print(popul_tot)
# print(crime)



# 히트맵
import seaborn as sns
sns.pairplot(crime_analysis, kind='reg')
plt.show()

# 피어슨 상관계수로 각 독립변수와 종속변수의 상관계수 구하기
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['CPI'])) # PearsonRResult(statistic=-0.8365563129796322, pvalue=0.00010123284951736509)
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['실업률'])) # PearsonRResult(statistic=-0.5535895526397691, pvalue=0.032280268638580974)
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['스트레스_인지율'])) # PearsonRResult(statistic=0.8655261708093139, pvalue=3.065014812312715e-05)
print(stats.pearsonr(crime_analysis['발생건수'], crime_analysis['인구수']))  # PearsonRResult(statistic=0.903434689304182, pvalue=3.915084633601483e-06)
# 이때, 4개의 변수 모두 0.05보다 작으므로 종속변수인 범죄건수(발생건수)에 유의한 영향을 미칠 것으로 간주할 수 있다.

# bartlett을 사용한 등분산 검정
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['CPI']))
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['실업률']))
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['스트레스_인지율']))
print(stats.bartlett(crime_analysis['발생건수'], crime_analysis['인구수']))
# 4개 변수 모두 pvalue가 0.05보다 작으므로 각 변수들이 통계적으로 유의하다.는 것을 알 수 있다.

# 독립성 검정 shapiro 사용하기
import statsmodels.formula.api as smf
lm_mul = smf.ols(formula='발생건수 ~ CPI+실업률+스트레스_인지율+인구수', data=crime_analysis).fit()
print(lm_mul.summary())
# Durbin-Watson:2.097 이므로 2에 가깝기 때문에 자기상관이 없다고 판단, 즉 독립성을 가정할 수 있다.

# 다중공선성
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(variance_inflation_factor(crime_analysis.values, 1))
print(variance_inflation_factor(crime_analysis.values, 2))
print(variance_inflation_factor(crime_analysis.values, 3))
print(variance_inflation_factor(crime_analysis.values, 4))
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(crime_analysis.values, i) for i in range(1, 5)]
print(vifdf)
#      vif_value
# 0   474.653839
# 1   904.303298
# 2  3109.624032
# 3   858.384493

# 위 결과를 종합적으로 보았을 때 88.1%의 모형 설명력을 가진 모델에 대해 4가지 변수들 모두 다중선형회귀분석을 실시하겠다.







