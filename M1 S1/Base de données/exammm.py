import pandas as pd

data = {}

#11
commandes = pd.DataFrame(data)
commandes.groupby(['annee','trimestre','mois']).agg({'port':['sum','mean','min','max']})

#12
df = pd.DataFrame(data)
df.pivot_table(index='Annee',columns='Zone',values='Temperature',aggfunc='mean')

#13
meteo01 = pd.DataFrame(data)
fm = meteo01[(meteo01['Mois']<3) & (meteo01['Annee']>1999)]
fm['avg_AnneeMois'] = fm.groupby(['Annee','Mois'])['Temperature'].transform('mean')
fm['avg_ZoneMois'] = fm.groupby(['Zone','Mois'])['Temperature'].transform('mean')

#14
df = pd.DataFrame(data)
df['sum_AnneeMois'] = df.groupby(['Annee','Mois'])['Precipitation'].transform('sum')
df['sum_CumulAnnee'] = df.groupby('Annee')['Precipitation'].cumsumm()

#15
patients = pd.DataFrame(data)
patients.groupby(['annee', 'gender']).agg({
    'alzheimer':['sum','min','max'],
    'heartfailure':['sum','min','max'],
    'cancer':['sum','min','max'],
    'stroke':['sum','min','max']
})

#16
patients = pd.DataFrame(data)
patients_hospitalises = pd.DataFrame(data)

resultat = patients.join(patients_hospitalises, how='inner')
resultat = resultat[['data_naissance','gender','data_admission','data_sortie','remboursement','franchise']]