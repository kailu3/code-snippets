# Figure 6
fig = go.Figure()

fig.add_trace(
   go.Scatter(
       x=np.arange(0.01, 1.00, 0.001),
       y=p_betas,
       mode='lines',
       name='Prior'
   )
)

fig.add_trace(
   go.Scatter(
       x=np.arange(0.01, 1.00, 0.001),
       y=zerozero,
       mode='lines',
       name='Posterior for x = 0, t_x = 0 (1995)'
   )
)

fig.add_trace(
   go.Scatter(
       x=np.arange(0.01, 1.00, 0.001),
       y=threethree,
       mode='lines',
       name='Posterior for x = 3, t_x = 3 (1998)'
   )
)


fig.update_layout(title='',
                  xaxis_title='p',
                  yaxis_title='f(p)',
                    xaxis = dict(
                    tickmode = 'linear',
                    tick0 = 0,
                    dtick = .25
                    ))
fig.show()


# Figure 7
fig = go.Figure()

fig.add_trace(
   go.Scatter(
       x=np.arange(0.01, 1.00, 0.001),
       y=theta_betas,
       mode='lines',
       name='Prior'
   )
)

fig.add_trace(
   go.Scatter(
       x=np.arange(0.01, 1.00, 0.001),
       y=threethree_theta,
       mode='lines',
       name='Posterior for x = 3, t_x = 3 (1998)'
   )
)

fig.add_trace(
   go.Scatter(
       x=np.arange(0.01, 1.00, 0.001),
       y=threesix_theta,
       mode='lines',
       name='Posterior for x = 3, t_x = 6 (2001)'
   )
)


fig.update_layout(title='',
                  xaxis_title='$\Theta$',
                  yaxis_title='f(p)',
                    xaxis = dict(
                    tickmode = 'linear',
                    tick0 = 0,
                    dtick = .25
                    ))
fig.show()