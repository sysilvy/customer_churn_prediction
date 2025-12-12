import plotly.graph_objects as go

def create_gauge_chart(probability):
  # determine color based on churn probability
  if probability < 0.3:
    color = 'green'
  elif probability < 0.6:
    color = 'yellow'
  else:
    color = 'red'

  # create a gauge chart
  fig = go.Figure(
    go.Indicator(
      mode="gauge+number",
      value=probability * 100,
      domain={
        'x': [0, 1],
        'y': [0, 1]},
      title={
        'text': "Churn Probability",
        'font': {
          'size': 24,
          'color': 'white'
        }
      },
      number={
        'font': {
          'size': 40,
          'color': 'white'
        }
      },
      gauge={
        'axis': {
          'range': [0, 100],
          'tickwidth': 1,
          'tickcolor': 'white'
        },
        'bar': {
          'color': color
        },
        'bgcolor': 'black',
        'borderwidth': 2,
        'bordercolor': 'white',
        'steps': [{
          'range': [0, 30],
          'color': 'rgba(0, 255, 0, 0.3)'
        }, {
          'range': [30, 60],
          'color': 'rgba(255, 255, 0, 0.3)'
        }, {
          'range': [60, 100],
          'color': 'rgba(255, 0, 0, 0.3)'
        }],
        'threshold': {
          'line': {
            'color': 'white',
            'width': 4
          },
          'thickness': 0.75,
          'value': 100
        }
      }
    )
  )

  # update chart layout
  fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                    font={'color': 'white'},
                   plot_bgcolor='rgba(0,0,0,0)',
                   width=400,
                   height=300,
                   margin=dict(l=20, r=20, b=50, t=20))
  return fig 


def create_model_probability_chart(probabilities):
  models = list(probabilities.keys())
  probs = list(probabilities.values())

  fig = go.Figure(data=[
    go.Bar(y=models,
          x=probs,
          orientation='h',
          text=[f'{p:.2%}' for p in probs],
          textposition='auto')
  ])

  fig.update_layout(title='Churn Probability by Model',
                   yaxis_title='Models',
                   xaxis_title='Probability',
                   yaxis={'autorange': 'reversed'},
                   height=400,
                   margin=dict(l=20, r=20, b=20, t=40))
  return fig


def create_percentile_chart(percentiles):
  metrics = list(percentiles.keys())
  values = [percentiles[m] for m in metrics]

  fig = go.Figure(data=[
    go.Bar(
      y=metrics,
      x=values,
      orientation='h',
      text=[f'{v:.0f}%' for v in values],
      textposition='auto',
      marker_color='lightblue'
    )
  ])

  fig.update_layout(
    title='Customer Percentiles',
    yaxis_title='Metric',
    xaxis_title='Percentile',
    xaxis=dict(range=[0, 100]),
    height=400,
    margin=dict(l=20, r=20, b=20, t=40),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': 'white'}
  )
  return fig