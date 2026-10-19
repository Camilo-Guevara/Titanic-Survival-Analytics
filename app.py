"""Titanic Survival Analytics | Streamlit | Cristian Camilo Guevara."""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
NAVY, CYAN, GOLD, PURPLE = '#071322', '#3be4e8', '#eac27a', '#a58aff'
PLOT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#d8e6f6', margin=dict(l=16,r=16,t=42,b=16))
PORTS = {'S':'Southampton', 'C':'Cherbourg', 'Q':'Queenstown'}

st.set_page_config(page_title='Titanic | Survival Analytics', page_icon='🚢', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp {background: radial-gradient(circle at 85% 0%,#192c4b 0%,#081526 36%,#050d19 100%);color:#e8f1ff}
[data-testid="stSidebar"] {background:#081527;border-right:1px solid #29415a}
h1,h2,h3 {letter-spacing:-.03em}
.hero {padding:30px 34px;border:1px solid #2c526d;border-radius:22px;background:linear-gradient(120deg,rgba(23,53,79,.94),rgba(12,20,41,.92));margin-bottom:20px}
.eyebrow {font-size:12px;letter-spacing:.23em;color:#3be4e8;font-weight:700}
.hero h1 {font-size:clamp(34px,4vw,60px);line-height:1.08;margin:10px 0;color:#f4f8ff}
.hero p {color:#b5c8db;font-size:16px;max-width:840px}
[data-testid="stMetric"] {background:rgba(15,39,59,.84);border:1px solid #25445c;padding:15px;border-radius:14px}
[data-testid="stMetricValue"] {color:#eac27a}
.stTabs [data-baseweb="tab-list"] {gap:12px;flex-wrap:wrap}
.stTabs [data-baseweb="tab"] {background:#10263b;border-radius:12px;color:#c6d7e7;padding:12px 16px}
.stTabs [aria-selected="true"] {background:#16445b!important;color:#7cf4f3!important}
div[data-testid="stPlotlyChart"] {border:1px solid #203b52;border-radius:16px;padding:7px;background:rgba(8,24,42,.6)}
.small-note {color:#a8bdd0;font-size:13px}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    original = pd.read_csv(BASE/'data'/'titanic_original.csv')
    processed = pd.read_csv(BASE/'data'/'titanic_procesado.csv')
    if len(original)!=891 or len(processed)!=891 or not original.PassengerId.equals(processed.PassengerId):
        raise ValueError('Los datasets no coinciden con la línea base validada de UTN.')
    return original, processed

@st.cache_resource
def train_model():
    # Important: split first; fit imputers and scaler only on training data.
    orig, _ = load_data()
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
    X, y = orig[features], orig['Survived']
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    numeric = ['Age','SibSp','Parch','Fare']
    categorical = ['Pclass','Sex','Embarked']
    prep = ColumnTransformer([
        ('numeric',Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('categorical',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical)
    ])
    model = Pipeline([('prep',prep),('clf',LogisticRegression(max_iter=2000,random_state=42))])
    model.fit(Xtr,ytr)
    preds = model.predict(Xte)
    prob = model.predict_proba(Xte)[:,1]
    return model, {'accuracy':accuracy_score(yte,preds),'auc':roc_auc_score(yte,prob),'test_n':len(yte),'cm':confusion_matrix(yte,preds).tolist()}

def plot(fig, key=None):
    fig.update_layout(**PLOT)
    st.plotly_chart(fig,use_container_width=True,key=key)

def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8-sig')

def percent(value):
    return f'{value:.1%}'

original, processed = load_data()
st.markdown('''<div class="hero"><div class="eyebrow">UTN · DATA SCIENCE · INTERACTIVE EXPERIENCE</div>
<h1>TITANIC <span style="color:#3be4e8">SURVIVAL</span> ANALYTICS</h1>
<p>891 historias. Un naufragio. Múltiples patrones para descubrir. Explorá la evidencia, auditá los datos y probá un modelo educativo de supervivencia.</p>
<div class="small-note">Cristian Camilo Guevara · Ciencia de Datos, Python y Análisis de Sistemas</div></div>''',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### ⚙️ Centro de control')
    st.caption('Estos filtros afectan las métricas y los gráficos del análisis exploratorio.')
    age_range = st.slider('Edad (años)',0,80,(0,80))
    classes = st.multiselect('Clase del pasaje',[1,2,3],default=[1,2,3],format_func=lambda x:f'{x}.ª clase')
    fare_range = st.slider('Tarifa del pasaje',0.0,float(np.ceil(processed.Fare.max())),(0.0,float(np.ceil(processed.Fare.max()))),step=1.0)
    sexes = st.multiselect('Sexo',['female','male'],default=['female','male'],format_func=lambda x:'Mujeres' if x=='female' else 'Hombres')
    ports = st.multiselect('Puerto de embarque',['S','C','Q'],default=['S','C','Q'],format_func=lambda x:PORTS[x])
    st.divider()
    st.caption('Fuente: dataset Titanic de la cátedra UTN · 891 registros. Los filtros usan Age procesada y Embarked imputado.')
    st.caption('Autor: Cristian Camilo Guevara · 2026')

filtered = processed.loc[(processed.Age.between(*age_range)) & processed.Pclass.isin(classes) & processed.Fare.between(*fare_range) & processed.Sex.isin(sexes) & processed.Embarked.isin(ports)].copy()
base_rate = processed.Survived.mean()
if filtered.empty:
    st.warning('No hay pasajeros con esa combinación de filtros. Ampliá los rangos en el panel lateral.')
    st.stop()

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric('Pasajeros',f'{len(filtered):,}',f'{len(filtered)/len(processed):.1%} del total')
m2.metric('Supervivencia',percent(filtered.Survived.mean()),f'{(filtered.Survived.mean()-base_rate)*100:+.1f} pp')
m3.metric('Sobrevivieron',int(filtered.Survived.sum()))
m4.metric('Tarifa media',f'{filtered.Fare.mean():.2f}')
m5.metric('Edad media',f'{filtered.Age.mean():.1f} años')
st.caption('Las métricas anteriores corresponden a la selección actual. La diferencia de supervivencia se expresa en puntos porcentuales (pp).')

tab_story,tab_etl,tab_eda,tab_model,tab_quality = st.tabs(['🌊 La historia','🧪 Laboratorio ETL','📊 Exploración EDA','🎯 Simulador ML','🛡️ Calidad y descargas'])

with tab_story:
    st.header('La historia detrás de los números')
    st.markdown('En la noche del **14 al 15 de abril de 1912**, el Titanic se hundió en el Atlántico Norte. Este dataset representa una **muestra de 891 pasajeros**, no la totalidad de las personas a bordo.')
    left,right = st.columns([1,1.2])
    with left:
        st.subheader('01 / ¿Quién tuvo más oportunidades?')
        rates = filtered.groupby('Sex',as_index=False).Survived.mean()
        rates['Sexo']=rates.Sex.map({'female':'Mujeres','male':'Hombres'})
        plot(px.bar(rates,x='Sexo',y='Survived',color='Sexo',color_discrete_map={'Mujeres':GOLD,'Hombres':CYAN},labels={'Survived':'Tasa de supervivencia'},range_y=[0,1],title='Supervivencia según sexo'))
    with right:
        st.subheader('02 / El peso de la clase')
        byclass=filtered.groupby('Pclass',as_index=False).Survived.mean()
        byclass['Clase']=byclass.Pclass.astype(str)+'ª'
        plot(px.bar(byclass,x='Clase',y='Survived',color='Clase',labels={'Survived':'Tasa de supervivencia'},range_y=[0,1],title='Supervivencia por clase'))
    st.info('Interpretación: sexo y clase se asocian fuertemente con la supervivencia, pero este análisis observacional no demuestra causalidad.')
    st.subheader('03 / El viaje de los pasajeros')
    st.caption('Sankey: puerto de embarque → clase → resultado. Los flujos corresponden a los pasajeros filtrados.')
    flow = filtered.copy()
    flow['Puerto']='Puerto: '+flow.Embarked.map(PORTS)
    flow['Clase']='Clase: '+flow.Pclass.astype(str)
    flow['Resultado']=np.where(flow.Survived==1,'Sobrevivió','No sobrevivió')
    nodes = ['Puerto: '+PORTS[x] for x in ['S','C','Q']]+['Clase: '+str(x) for x in [1,2,3]]+['Sobrevivió','No sobrevivió']
    idx={n:i for i,n in enumerate(nodes)}
    links=[]
    for a,b in [('Puerto','Clase'),('Clase','Resultado')]:
        for (src,dst),count in flow.groupby([a,b]).size().items():
            links.append((idx[src],idx[dst],int(count)))
    fig=go.Figure(go.Sankey(node=dict(label=nodes,pad=22,thickness=16,color=[CYAN]*3+[PURPLE]*3+[GOLD,'#64748b']),link=dict(source=[x[0] for x in links],target=[x[1] for x in links],value=[x[2] for x in links],color='rgba(72,178,200,.28)')))
    fig.update_layout(height=420,title='Rutas hacia la supervivencia')
    plot(fig)
    with st.expander('¿Qué representa este diagrama?'):
        st.write('Cada banda representa una cantidad de pasajeros. El flujo se divide en dos niveles: puerto → clase y clase → supervivencia. Es una visualización agregada de frecuencias, no un seguimiento causal individual.')

with tab_etl:
    st.header('🧪 Laboratorio ETL · De datos crudos a información útil')
    st.markdown('**Explorá visualmente qué se corrigió, qué se conservó y por qué.** Los experimentos de esta pestaña no alteran los archivos originales.')
    st.info('💡 **Cómo leer la comparación:** amarillo = celdas vacías antes de limpiar; cian = celdas vacías después. Una barra que permanece alta, como Cabina, indica una decisión consciente de **no inventar información**.')
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Pasajeros conservados',f'{len(processed)} / {len(original)}')
    k2.metric('Edades completadas',int(original.Age.isna().sum()-processed.Age.isna().sum()))
    k3.metric('Embarques completados',int(original.Embarked.isna().sum()-processed.Embarked.isna().sum()))
    k4.metric('Variables nuevas',len(processed.columns)-len(original.columns))

    st.subheader('01 / Radiografía de la limpieza')
    focus_cols=['Age','Cabin','Embarked']
    names={'Age':'Edad','Cabin':'Cabina','Embarked':'Embarque'}
    comparison=pd.DataFrame({'Variable':[names[x] for x in focus_cols],
        'Antes':[int(original[x].isna().sum()) for x in focus_cols],
        'Después':[int(processed[x].isna().sum()) for x in focus_cols]})
    chart_col,guide_col=st.columns([1.7,1])
    with chart_col:
        long=comparison.melt(id_vars='Variable',var_name='Momento',value_name='Celdas vacías')
        fig=px.bar(long,x='Variable',y='Celdas vacías',color='Momento',barmode='group',text='Celdas vacías',
                   color_discrete_map={'Antes':GOLD,'Después':CYAN},category_orders={'Variable':['Edad','Cabina','Embarque']},
                   title='Antes y después · Solo columnas con nulos originales')
        fig.update_traces(textposition='outside',hovertemplate='%{x}<br>%{fullData.name}: %{y} valores faltantes<extra></extra>')
        fig.update_layout(yaxis_range=[0,760],height=380,legend_title_text='Versión')
        plot(fig,key='etl_before_after_focus')
    with guide_col:
        st.markdown('**Interpretación rápida**')
        st.success('**Edad:** 177 → 0. Se imputaron valores estimados, no edades históricas verificadas.')
        st.success('**Embarque:** 2 → 0. Se usó la categoría más frecuente (S).')
        st.warning('**Cabina:** 687 → 687. Se mantuvo desconocida para evitar fabricar datos.')
    st.dataframe(comparison.assign(**{'Cambio':comparison['Después']-comparison['Antes'],
                                    '% faltante antes':(comparison['Antes']/len(original)*100).round(1),
                                    '% faltante después':(comparison['Después']/len(processed)*100).round(1)}),
                 hide_index=True,use_container_width=True)

    st.subheader('02 / Imagen interactiva: mapa de los datos ausentes')
    st.caption('Pasá el cursor por cada celda para descubrir si el dato faltaba. Elegí qué variable inspeccionar y cuántos pasajeros mostrar.')
    csel,cnum=st.columns([2,1])
    with csel:
        selected_missing=st.multiselect('Variables a inspeccionar',['Age','Cabin','Embarked'],default=['Age','Cabin','Embarked'],
                                        format_func=lambda x:names[x],key='etl_missing_select')
    with cnum:
        nrows=st.slider('Pasajeros visualizados',20,200,80,20,key='etl_nrows')
    if selected_missing:
        compare_version=st.radio('Versión del mapa',['Antes: datos crudos','Después: datos procesados'],horizontal=True,key='etl_heat_version')
        source=original if compare_version.startswith('Antes') else processed
        mask=source[selected_missing].iloc[:nrows].isna().astype(int).T
        fig=go.Figure(go.Heatmap(z=mask.to_numpy(),x=source.PassengerId.iloc[:nrows].tolist(),
             y=[names[c] for c in selected_missing],zmin=0,zmax=1,
             colorscale=[[0,'#17445b'],[0.499,'#17445b'],[0.5,GOLD],[1,GOLD]],
             customdata=np.array([['FALTANTE' if v else 'PRESENTE' for v in row] for row in mask.to_numpy()]),
             hovertemplate='Pasajero ID %{x}<br>Variable: %{y}<br>Estado: %{customdata}<extra></extra>',
             colorbar=dict(tickvals=[0,1],ticktext=['Presente','Faltante'],title='Estado')))
        fig.update_layout(title='Cada cuadrícula es una celda del dataset',height=max(250,110+75*len(selected_missing)),
                          xaxis_title='ID del pasajero',yaxis_title='Variable')
        plot(fig,key='etl_interactive_missing')
    else:
        st.warning('Seleccioná al menos una variable para mostrar el mapa.')

    st.subheader('03 / Microscopio de pasajeros: ficha original vs. ficha procesada')
    st.caption('Seleccioná un pasajero con un dato originalmente faltante y observá el cambio real. Los valores estimados están señalados.')
    cases=original.loc[original[['Age','Cabin','Embarked']].isna().any(axis=1),'PassengerId'].tolist()
    chosen=st.selectbox('Pasajero (PassengerId)',cases,index=0,key='etl_passenger_id')
    before=original.loc[original.PassengerId.eq(chosen)].iloc[0]
    after=processed.loc[processed.PassengerId.eq(chosen)].iloc[0]
    rows=[]
    for col in ['Age','Cabin','Embarked']:
        old=before[col]; updated=after[col]
        changed=pd.isna(old) and pd.notna(updated)
        rows.append({'Campo':names[col], 'Antes (crudo)':'❓ Sin dato' if pd.isna(old) else str(old),
                     'Después (procesado)':'❓ Sin dato' if pd.isna(updated) else str(updated),
                     'Interpretación':'🟡 Imputado: valor estimado' if changed else ('🔒 Conservado como desconocido' if pd.isna(old) else '✔ Sin cambio')})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    if pd.isna(before.Age):
        st.caption('La edad procesada de este pasajero es una **estimación**; no representa una edad verificada en registros históricos.')

    st.subheader('04 / Simulador ETL: ¿qué pasaría con otra estrategia?')
    strategy=st.selectbox('Método para tratar edades faltantes',
        ['Etapa 2: mediana por grupos','Mediana global','Media global','Eliminar filas sin edad'],key='etl_strategy')
    experiment=original.copy()
    if strategy=='Mediana global':
        experiment['Age']=experiment.Age.fillna(experiment.Age.median())
    elif strategy=='Media global':
        experiment['Age']=experiment.Age.fillna(experiment.Age.mean())
    elif strategy=='Eliminar filas sin edad':
        experiment=experiment.dropna(subset=['Age']).copy()
    else:
        experiment['Age']=processed['Age'].to_numpy()
    e1,e2,e3,e4=st.columns(4)
    e1.metric('Pasajeros resultantes',len(experiment),len(experiment)-len(original))
    e2.metric('Edad promedio',f'{experiment.Age.mean():.2f}')
    e3.metric('Edades faltantes',int(experiment.Age.isna().sum()))
    e4.metric('Supervivencia en muestra',f'{experiment.Survived.mean():.1%}')
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=original.Age.dropna(),name='Edades observadas',opacity=.65,marker_color=GOLD,nbinsx=25))
    fig.add_trace(go.Histogram(x=experiment.Age.dropna(),name='Resultado del experimento',opacity=.60,marker_color=CYAN,nbinsx=25))
    fig.update_layout(barmode='overlay',title='Imagen interactiva · Cómo cambia la distribución de edades',
                      xaxis_title='Edad (años)',yaxis_title='Número de pasajeros',height=400)
    plot(fig,key='etl_age_experiment')
    st.caption('Los histogramas superpuestos no son grupos independientes: comparten pasajeros. Cambiar de estrategia NO modifica el dataset oficial procesado ni el modelo ML.')
    if strategy=='Eliminar filas sin edad':
        st.warning('Eliminar las filas reduce la muestra y puede introducir sesgos. La tasa de supervivencia del subconjunto puede cambiar por selección de registros.')

    st.subheader('05 / Ingeniería de características: del registro a nuevas pistas')
    feat=processed[['PassengerId','Name','Title','SibSp','Parch','FamilySize','IsAlone','Cabin','CabinKnown','Deck','AgeWasMissing']].copy()
    st.dataframe(feat.head(30),use_container_width=True,hide_index=True)
    fa,fb=st.columns(2)
    with fa:
        family=processed.groupby('FamilySize').size().reset_index(name='Pasajeros')
        plot(px.bar(family,x='FamilySize',y='Pasajeros',color='Pasajeros',
                    color_continuous_scale=['#1c536c',CYAN],title='Imagen interactiva · Tamaño del grupo familiar'),key='etl_family')
    with fb:
        titlecounts=processed.Title.fillna('Desconocido').value_counts().head(8).reset_index()
        titlecounts.columns=['Título','Pasajeros']
        plot(px.bar(titlecounts,x='Pasajeros',y='Título',orientation='h',color='Pasajeros',
                    color_continuous_scale=['#4b3b79',PURPLE],title='Imagen interactiva · Títulos extraídos de los nombres'),key='etl_titles')
    with st.expander('📘 Ver fórmulas y decisiones metodológicas'):
        st.code('FamilySize = SibSp + Parch + 1\nIsAlone = (FamilySize == 1).astype(int)\nTitle = Name.str.extract(r",\\s*([^.]*)\\.")\nCabinKnown = Cabin.notna().astype(int)',language='python')
        st.write('El valor de Cabin se mantiene vacío cuando es desconocido; CabinKnown y Deck permiten estudiar la disponibilidad de esa información. El modelo ML usa OneHotEncoder ajustado dentro del conjunto de entrenamiento.')
    st.download_button('⬇ Descargar tabla comparativa de nulos',csv_bytes(comparison),'titanic_comparacion_nulos.csv','text/csv',key='etl_download_comparison')

with tab_eda:
    st.header('Exploración multidimensional')
    st.write('Todos los gráficos de esta pestaña se recalculan al cambiar los filtros del panel lateral.')
    a,b=st.columns(2)
    with a:
        plot(px.violin(filtered,x='Pclass',y='Fare',color=filtered.Survived.map({0:'No sobrevivió',1:'Sobrevivió'}),box=True,points=False,color_discrete_map={'Sobrevivió':GOLD,'No sobrevivió':CYAN},labels={'Pclass':'Clase','Fare':'Tarifa','color':'Resultado'},title='Distribución de tarifas por clase y supervivencia'))
    with b:
        plot(px.box(filtered,x=filtered.Survived.map({0:'No sobrevivió',1:'Sobrevivió'}),y='Age',color='Sex',color_discrete_map={'female':GOLD,'male':CYAN},labels={'x':'Resultado','Age':'Edad','Sex':'Sexo'},title='Edad y supervivencia según sexo'))
    st.subheader('Pirámide de edades: supervivencia')
    pyr=filtered.copy()
    pyr['Tramo']=pd.cut(pyr.Age,bins=[0,10,20,30,40,50,60,70,81],right=False,labels=['0–9','10–19','20–29','30–39','40–49','50–59','60–69','70–80'])
    pyramid=pyr.groupby(['Tramo','Survived'],observed=True).size().reset_index(name='Pasajeros')
    pyramid['Estado']=pyramid.Survived.map({0:'No sobrevivió',1:'Sobrevivió'})
    pyramid['Valor']=np.where(pyramid.Survived==0,-pyramid.Pasajeros,pyramid.Pasajeros)
    fig=px.bar(pyramid,x='Valor',y='Tramo',orientation='h',color='Estado',color_discrete_map={'Sobrevivió':GOLD,'No sobrevivió':CYAN},title='No sobrevivieron ← | → Sobrevivieron',labels={'Valor':'Pasajeros (magnitud)','Tramo':'Edad'})
    fig.update_layout(barmode='relative')
    plot(fig)
    st.caption('La pirámide separa por estado de supervivencia, no por sexo. Los valores negativos son una convención gráfica, no cantidades negativas.')
    st.subheader('Correlaciones numéricas')
    numeric=filtered.select_dtypes(include='number').drop(columns=['PassengerId'],errors='ignore')
    if len(numeric)>2:
        corr=numeric.corr(numeric_only=True)
        fig=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale='RdBu_r',aspect='auto',text_auto='.2f',title='Correlación de Pearson (datos filtrados)')
        fig.update_layout(height=650)
        plot(fig)
    st.caption('Las correlaciones no equivalen a causalidad; las variables binarias y derivadas requieren interpretación contextual.')
    with st.expander('Ver tabla de pasajeros filtrados'):
        st.dataframe(filtered,use_container_width=True,hide_index=True)
        st.download_button('⬇ Descargar selección filtrada',csv_bytes(filtered),'titanic_filtrado.csv','text/csv')

with tab_model:
    st.header('Simulador de supervivencia · Machine Learning')
    st.warning('Modelo educativo basado en una muestra histórica. La probabilidad es una estimación estadística, NO una probabilidad histórica exacta ni una predicción individual fiable.')
    model, metrics=train_model()
    aa,bb,cc=st.columns(3)
    aa.metric('Accuracy de prueba',f"{metrics['accuracy']:.1%}")
    bb.metric('ROC-AUC de prueba',f"{metrics['auc']:.3f}")
    cc.metric('Muestra de prueba',metrics['test_n'])
    st.caption('Regresión logística; división estratificada 75/25; random_state=42. Imputación, estandarización y OneHotEncoder se ajustan solo con entrenamiento para reducir fuga de información.')
    with st.form('survival_form'):
        c1,c2=st.columns(2)
        with c1:
            cls=st.selectbox('¿En qué clase viajarías?', [1,2,3],format_func=lambda x:f'{x}.ª clase')
            sex=st.selectbox('Sexo registrado',['female','male'],format_func=lambda x:'Mujer' if x=='female' else 'Hombre')
            age=st.slider('Edad',0,80,28)
            port=st.selectbox('Puerto de embarque',['S','C','Q'],format_func=lambda x:PORTS[x])
        with c2:
            sibsp=st.number_input('Hermanos/as o cónyuges a bordo',min_value=0,max_value=8,value=0)
            parch=st.number_input('Padres/madres o hijos/as a bordo',min_value=0,max_value=6,value=0)
            fare=st.number_input('Tarifa del pasaje',min_value=0.0,max_value=550.0,value=20.0,step=5.0)
            st.caption('El tamaño familiar se deriva de los dos campos anteriores.')
        submitted=st.form_submit_button('🚢 Calcular mi escenario',use_container_width=True)
    if submitted:
        row=pd.DataFrame([{'Pclass':cls,'Sex':sex,'Age':age,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':port}])
        probability=float(model.predict_proba(row)[0,1])
        left,right=st.columns([1,1])
        with left:
            gauge=go.Figure(go.Indicator(mode='gauge+number',value=probability*100,number={'suffix':'%'},title={'text':'Probabilidad estimada'},gauge={'axis':{'range':[0,100]},'bar':{'color':CYAN},'steps':[{'range':[0,40],'color':'#20364b'},{'range':[40,70],'color':'#244d60'},{'range':[70,100],'color':'#326a75'}]}))
            gauge.update_layout(height=310)
            plot(gauge)
        with right:
            st.subheader('Tu escenario a bordo')
            st.write(f'Viajarías en **{cls}.ª clase**, desde **{PORTS[port]}**, con **{sibsp+parch} familiares** registrados.')
            st.write('El modelo estima una probabilidad relativamente '+('alta' if probability>=.5 else 'baja')+' de supervivencia para este perfil.')
            st.info('El resultado puede cambiar con la selección de variables, el conjunto de entrenamiento y el algoritmo. No permite reconstruir el destino real de una persona.')
    with st.expander('Metodología, evaluación y limitaciones'):
        st.write('Se entrenó una regresión logística con Pclass, Sex, Age, SibSp, Parch, Fare y Embarked. Los datos se dividen antes de imputar. El modelo no utiliza Survived para transformar variables. Se evalúa en un conjunto reservado. La accuracy y ROC-AUC son métricas de esta división, no garantías de generalización.')
        st.write('Sesgos históricos, datos faltantes y representatividad limitada impiden interpretar las probabilidades como hechos. El simulador es para aprendizaje de ML.')
        st.write('Matriz de confusión (filas: real 0/1; columnas: predicho 0/1):')
        st.dataframe(pd.DataFrame(metrics['cm'],index=['Real 0','Real 1'],columns=['Predicho 0','Predicho 1']))

with tab_quality:
    st.header('Auditoría, trazabilidad y exportación')
    summary=pd.DataFrame({'Variable':original.columns,'Nulos original':[original[x].isna().sum() for x in original.columns],'Nulos procesado':[processed[x].isna().sum() for x in original.columns]})
    summary['% nulos original']=(summary['Nulos original']/len(original)*100).round(2)
    st.dataframe(summary,use_container_width=True,hide_index=True)
    q1,q2,q3=st.columns(3)
    q1.metric('Duplicados originales',int(original.duplicated().sum()))
    q2.metric('Edad imputada',int(original.Age.isna().sum()-processed.Age.isna().sum()))
    q3.metric('Cabinas sin dato',int(processed.Cabin.isna().sum()))
    st.subheader('Matriz de ausencia: muestra completa')
    mask=original[['Age','Cabin','Embarked']].isna().astype(int)
    fig=px.imshow(mask.T,aspect='auto',color_continuous_scale=[[0,'#142c44'],[1,GOLD]],zmin=0,zmax=1,labels={'x':'Índice del pasajero','y':'Variable','color':'Faltante'},title='Presencia / ausencia de datos (891 pasajeros)')
    fig.update_layout(height=270)
    plot(fig)
    st.markdown('**Decisiones documentadas:** Age imputada, Embarked imputado, Cabin conservada como faltante; indicadores AgeWasMissing y CabinKnown; outliers de Fare señalados pero no eliminados.')
    st.subheader('Descargas')
    d1,d2,d3=st.columns(3)
    d1.download_button('⬇ Dataset original',csv_bytes(original),'titanic_original.csv','text/csv',use_container_width=True)
    d2.download_button('⬇ Dataset procesado',csv_bytes(processed),'titanic_procesado.csv','text/csv',use_container_width=True)
    d3.download_button('⬇ Datos filtrados',csv_bytes(filtered),'titanic_filtrado.csv','text/csv',use_container_width=True)
    st.download_button('⬇ Reporte de calidad CSV',csv_bytes(summary),'titanic_calidad.csv','text/csv')
    st.caption('Los archivos originales y procesados son independientes; los experimentos ETL no modifican el origen.')

st.divider()
st.caption('Titanic Survival Analytics · Etapa 4 · Cristian Camilo Guevara · Proyecto académico UTN y portafolio profesional · 2026')