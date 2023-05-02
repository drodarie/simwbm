: Adaptive Exponential Integrate-and-Fire model with beta postsynaptic receptors
: Brette R. and Gerstner W. (2005), Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity, J. Neurophysiol. 94: 3637 - 3642.
: Implemented by Dimitri RODARIE, EPFL/Blue Brain Project, 2019


NEURON {
THREADSAFE
    POINT_PROCESS AdEx
    RANGE V_reset, t_ref, E_L, g_L, C_m, V_peak, a, b, tau_w, Delta_T, V_th, I_e, V_M, w, nb_receptors, nb_synapses, current_stimsize, spike
    POINTER E_rev, tau_rise, tau_decay, dg_in, g_in, g0, I_stim
    POINTER weights, tau_recs, tau_facs, Us, t_lastspikes, xs, us
    POINTER k1g, k1dg, k2g, k2dg, k3g, k3dg, k4g, k4dg, k5dg, k5g, k6g, k6dg, tmpdg, tmpg, dgs, gs
}

VERBATIM
extern int ifarg(int iarg);
#ifndef CORENEURON_BUILD
extern double* vector_vec(void* vv);
extern void* vector_new1(int _i);
extern int vector_capacity(void* vv);
#endif
ENDVERBATIM

UNITS {
    (mV) = (millivolt)
    (pA) = (picoampere)
    (nS) = (nanosiemens)
    (pF) = (picofarad)
}

PARAMETER {
    V_M  = -80.0 (mV)
    E_L = -50 (mV)
    g_L = 30.0 (nS)
    C_m = 281.0 (pF)
    I_e = 0.0 (pA)
    t_ref = 0.0 (ms)
    V_reset = -60 (mV)
    V_th = -48 (mV)
    V_peak = 0.0 (mV)
    Delta_T = 0.5 (mV)

    a = 4.0 (nS)
    b = 80.5 (pA)
    tau_w = 144.0 (ms)
    nb_receptors = 0 (1)
    current_stimsize = 0 (1)
    nb_synapses = 0 (1)
    epsilon = 1e-6 (1)
}

ASSIGNED {
    iterator (1)
    I_stim(pA)
    loc_i_stim(pA)
    dt (ms)
    w  (pA)
    g0 (nS) : Peak conductance for postsyn receptor
    t0 (ms)
    R (1)
    force_integration (1)
    flag_update_time (1)
    refractory (1)

    E_rev (mV)
    tau_rise (ms)
    tau_decay (ms)
    dg_in (nS)
    g_in (nS)

    weights (1)
    tau_recs (ms)
    tau_facs (ms)
    Us (1)
    t_lastspikes (ms)
    xs (1)
    us (1)
    k1g (1)
    k1dg (1)
    k2g (1)
    k2dg (1)
    k3g (1)
    k3dg (1)
    k4g (1)
    k4dg (1)
    k5g (1)
    k5dg (1)
    k6g (1)
    k6dg (1)
    tmpdg (1) 
    tmpg (1) 
    dgs (1)
    gs (1)
    spike (1)
}

INITIAL {
    w = 0
    spike = 0
    iterator = -1
    loc_i_stim = 0
    if ( Delta_T <= 0.0 )
    {
      V_peak = V_th : same as IAF dynamics for spikes if Delta_T == 0.
    }
    VERBATIM
      _p_g0 = (double*) vector_new1((int)nb_receptors);
      int j=0;
      if (nb_receptors>0)
      {
        double *tau_d = vector_vec(_p_tau_decay), *tau_r = vector_vec(_p_tau_rise), *g0_double = vector_vec(_p_g0);
        while (j<nb_receptors)
        {
          double t_p, denom1 = tau_d[j] - tau_r[j], denom2 = 0.0;
          if ( denom1 != 0 )
          {
            t_p = tau_d[j] * tau_r[j] * log(tau_d[j] / tau_r[j]) / denom1;
            denom2 = exp( -t_p / tau_d[j]) - exp( -t_p / tau_r[j] );
          }
          if ( denom2 == 0 )
          {
            g0_double[j] = 2.71828182845904523536028747135 / tau_d[j];
          }
          else
          {
            g0_double[j] =  (1.0 / tau_r[j] - 1.0 / tau_d[j] ) / denom2;
          }
          j= j+1;
        }
      }
      _p_k1g = vector_new1((int)nb_receptors);
      _p_k1dg = vector_new1((int)nb_receptors);
      _p_k2g = vector_new1((int)nb_receptors);
      _p_k2dg = vector_new1((int)nb_receptors);
      _p_k3g = vector_new1((int)nb_receptors);
      _p_k3dg = vector_new1((int)nb_receptors);
      _p_k4g = vector_new1((int)nb_receptors);
      _p_k4dg = vector_new1((int)nb_receptors);
      _p_k5dg = vector_new1((int)nb_receptors);
      _p_k5g = vector_new1((int)nb_receptors);
      _p_k6g = vector_new1((int)nb_receptors);
      _p_k6dg = vector_new1((int)nb_receptors);
      _p_dg_in = vector_new1((int)nb_receptors);
      _p_g_in = vector_new1((int)nb_receptors);
      _p_tmpdg = vector_new1((int)nb_receptors);
      _p_tmpg =vector_new1((int)nb_receptors); 
      _p_dgs =vector_new1((int)nb_receptors); 
      _p_gs =vector_new1((int)nb_receptors);
      double *dg_in_double = vector_vec(_p_dg_in), *g_in_double = vector_vec(_p_g_in);
      for (j=0; j<(int)nb_receptors; j++)
      {
        dg_in_double[j] = 0.0;
        g_in_double[j] = 0.0;
      }
    ENDVERBATIM

    t0 = t
    force_integration=0
    flag_update_time=0
    refractory = 0
    :printf("%f %f %f %f %f %f %f %f %f %f %f \n", a, b, E_L, g_L, C_m, I_e, t_ref, V_reset, V_th, V_peak, Delta_T )

}
VERBATIM
void setArray(void* v1, void* v2)
{
  int j=0;
  u_int32_t dsize = (u_int32_t)vector_capacity(v1);
  double *v1_double = vector_vec(v1), *v2_double = vector_vec(v2);
  for (j=0; j<dsize; j++)
  {
    v1_double[j] = v2_double[j];
  }
}

void mul1(double factor, void* v1){
  int iInt;
  u_int32_t dsize = (u_int32_t)vector_capacity(v1);
  double *v1_double = vector_vec(v1);
  for (iInt = 0; iInt < dsize; ++iInt)
  {
     v1_double[iInt] *= factor;
  }
}

double RKV(_threadargsproto_, double vin, void* gein, double uin) {
    double I_syn=0, Vsp;
    int j=0;

    u_int32_t dsize = (u_int32_t)vector_capacity(gein);
    double *gein_double = vector_vec(gein);
    if (dsize>0)
    {
      double *local_erev = vector_vec(_p_E_rev);
      for (j = 0; j < dsize; j++)
      {
        I_syn = I_syn + gein_double[j]*( local_erev[j] - vin );
      }
    }

    if (Delta_T <= 0) // Back to iaf dynamics
    {
      Vsp = 0;
    }
    else
    {
      Vsp = Delta_T * exp( (vin - V_th )/Delta_T );
    }
    return (-g_L * ( vin - E_L - Vsp ) + I_syn + loc_i_stim - uin + I_e ) / C_m;
}

void RKDG (_threadargsproto_, void* recept_array, void* dgin)
{
    if (nb_receptors>0)
    {
      u_int32_t dsize = (u_int32_t)vector_capacity(dgin);
      double* result = vector_vec(recept_array), *dgin_double = vector_vec(dgin), *tau_rise_double = vector_vec(_p_tau_rise);

      for (int j = 0; j < dsize; j++)
      {
        result[j] = -dgin_double[j] / tau_rise_double[j];
      }
    }
}

void RKG (_threadargsproto_, void* recept_array, void* dgin, void* gein)
{
  if (nb_receptors>0)
  {
    u_int32_t dsize = (u_int32_t)vector_capacity(dgin);
    double* result = vector_vec(recept_array), *dgin_double = vector_vec(dgin), *gein_double = vector_vec(gein), *tau_rise_double = vector_vec(_p_tau_rise), *tau_decay_double = vector_vec(_p_tau_decay);

    for (int j = 0; j < dsize; j++)
    {
      result[j] = dgin_double[j] - gein_double[j]/tau_decay_double[j];
    }
  }
}
ENDVERBATIM

FUNCTION RKW (vin, uin) {
  RKW = ( a * (vin - E_L) - uin ) / tau_w
}

FUNCTION ADVANCE_RK (integration_step) {
  VERBATIM
  int j;
  double tmpv, tmpw, k1v, k1w, k2v, k2w, k3v, k3w, k4v, k4w, vs, ws, delta, k5v, k5w, k6v, k6w, screwu, norm;
  double *tmpdg_double = vector_vec(_p_tmpdg), *tmpg_double =vector_vec(_p_tmpg), *dgs_double =vector_vec(_p_dgs), *gs_double =vector_vec(_p_gs);
  double* dg_in_double = vector_vec(_p_dg_in), *g_in_double = vector_vec(_p_g_in);

  force_integration = 0;
  if ( V_M < V_peak ) { tmpv = V_M; }
  else { tmpv = V_peak; }

  setArray(_p_tmpdg, _p_dg_in);
  setArray( _p_tmpg,  _p_g_in);
  tmpw = w;

  if ( _lintegration_step < 1e-5 ) {
    _lintegration_step = 1e-5;
    force_integration = 1;
    flag_update_time = 1;
  }

  k1v = _lintegration_step*RKV(_threadargs_, tmpv, _p_tmpg, tmpw);
  RKDG(_threadargs_, _p_k1dg, _p_tmpdg);
  mul1(_lintegration_step, _p_k1dg);
  RKG(_threadargs_,_p_k1g, _p_tmpdg, _p_tmpg);
  mul1(_lintegration_step, _p_k1g);
  k1w = _lintegration_step*RKW(_threadargs_,tmpv, tmpw);

  tmpv = V_M +  ( 1.0 / 4.0 )*k1v;

  double * k1dg_double = vector_vec(_p_k1dg), *k1g_double = vector_vec(_p_k1g);
  for (j=0; j<nb_receptors; j++)
  {
    tmpdg_double[j] = dg_in_double[j] + ( 1.0 / 4.0 )*k1dg_double[j];
     tmpg_double[j] =  g_in_double[j] + ( 1.0 / 4.0 )* k1g_double[j];
  }

  tmpw = w + ( 1.0 / 4.0 )*k1w;

  if ( tmpv > V_peak ) { tmpv = V_peak; }

  k2v = _lintegration_step*RKV(_threadargs_, tmpv, _p_tmpg, tmpw);
  RKDG(_threadargs_, _p_k2dg, _p_tmpdg);
  mul1(_lintegration_step, _p_k2dg);
  RKG(_threadargs_, _p_k2g, _p_tmpdg, _p_tmpg);
  mul1(_lintegration_step, _p_k2g);
  k2w = _lintegration_step*RKW(_threadargs_, tmpv, tmpw);

  tmpv  = V_M  + ( 3.0/32.0 )* k1v  + (9.0/32.0 )*k2v;

  double * k2dg_double = vector_vec(_p_k2dg), *k2g_double = vector_vec(_p_k2g);
  for (j=0; j<nb_receptors; j++)
  {
    tmpdg_double[j] = dg_in_double[j] + ( 3.0 / 32.0 )*k1dg_double[j] + (9.0/32.0 )*k2dg_double[j];
     tmpg_double[j] =  g_in_double[j] + ( 3.0 / 32.0 )* k1g_double[j] + (9.0/32.0 )* k2g_double[j];
  }

  tmpw  = w  + ( 3.0/32.0 )* k1w  + (9.0/32.0 )*k2w;

  if ( tmpv > V_peak ) { tmpv = V_peak; }

  k3v = _lintegration_step*RKV(_threadargs_, tmpv, _p_tmpg, tmpw);
  RKDG(_threadargs_, _p_k3dg, _p_tmpdg);
  mul1(_lintegration_step, _p_k3dg);
  RKG(_threadargs_, _p_k3g, _p_tmpdg, _p_tmpg);
  mul1(_lintegration_step, _p_k3g);
  k3w = _lintegration_step*RKW(_threadargs_, tmpv, tmpw);

  tmpv = V_M + (1932.0/2197.0)*k1v - (7200.0/2197.0)*k2v + ( 7296.0 / 2197.0 )*k3v;

  double * k3dg_double = vector_vec(_p_k3dg), *k3g_double = vector_vec(_p_k3g);
  for (j=0; j<nb_receptors; j++)
  {
    tmpdg_double[j] = dg_in_double[j] + ( 1932.0/2197.0 )*k1dg_double[j] - (7200.0/2197.0 )*k2dg_double[j] + ( 7296.0 / 2197.0 )*k3dg_double[j];
     tmpg_double[j] =  g_in_double[j] + ( 1932.0/2197.0 )* k1g_double[j] - (7200.0/2197.0 )* k2g_double[j] + ( 7296.0 / 2197.0 )* k3g_double[j];
  }

  tmpw = w + (1932.0/2197.0)*k1w - (7200.0/2197.0)*k2w + ( 7296.0 / 2197.0 )*k3w;

  if ( tmpv > V_peak ) { tmpv = V_peak; }

  k4v = _lintegration_step*RKV(_threadargs_, tmpv, _p_tmpg, tmpw);
  RKDG(_threadargs_, _p_k4dg, _p_tmpdg);
  mul1(_lintegration_step, _p_k4dg);
  RKG(_threadargs_, _p_k4g, _p_tmpdg, _p_tmpg);
  mul1(_lintegration_step, _p_k4g);
  k4w = _lintegration_step*RKW(_threadargs_, tmpv, tmpw);

  tmpv  = V_M  + (439.0 / 216.0) * k1v  - 8.0 * k2v   + ( 3680.0 / 513.0 ) * k3v - ( 845.0 / 4104.0 )*k4v;

  double * k4dg_double = vector_vec(_p_k4dg), *k4g_double = vector_vec(_p_k4g);
  for (j=0; j<nb_receptors; j++)
  {
    tmpdg_double[j] = dg_in_double[j] + ( 439.0 / 216.0 )*k1dg_double[j] - 8.0*k2dg_double[j] + ( 3680.0 / 513.0 )*k3dg_double[j] - ( 845.0 / 4104.0 )*k4dg_double[j];
     tmpg_double[j] =  g_in_double[j] + ( 439.0 / 216.0 )* k1g_double[j] - 8.0* k2g_double[j] + ( 3680.0 / 513.0 )* k3g_double[j] - ( 845.0 / 4104.0 )* k4g_double[j];
  }

  tmpw  = w  + (439.0 / 216.0) * k1w  - 8.0 * k2w   + ( 3680.0 / 513.0 ) * k3w   - ( 845.0 / 4104.0 )*k4w;

  if ( tmpv > V_peak ) { tmpv = V_peak; }

  k5v = _lintegration_step*RKV(_threadargs_, tmpv, _p_tmpg, tmpw);
  RKDG(_threadargs_, _p_k5dg, _p_tmpdg);
  mul1(_lintegration_step, _p_k5dg);
  RKG(_threadargs_, _p_k5g, _p_tmpdg, _p_tmpg);
  mul1(_lintegration_step, _p_k5g);
  k5w = _lintegration_step*RKW(_threadargs_, tmpv, tmpw);

  tmpv  = V_M  - (8.0/27.0)*k1v  + 2.0*k2v  - (3544.0/2565.0)*k3v  + (1859.0/4104.0)*k4v  - ( 11.0 / 40.0 )*k5v;

  double * k5dg_double = vector_vec(_p_k5dg), *k5g_double = vector_vec(_p_k5g);
  for (j=0; j<nb_receptors; j++)
  {
    tmpdg_double[j] = dg_in_double[j] - ( 8.0/27.0 )*k1dg_double[j] + 2.0*k2dg_double[j] - ( 3544.0/2565.0 )*k3dg_double[j] + ( 1859.0/4104.0 )*k4dg_double[j] - ( 11.0 / 40.0 )*k5dg_double[j];
     tmpg_double[j] =  g_in_double[j] - ( 8.0/27.0 )* k1g_double[j] + 2.0* k2g_double[j] - ( 3544.0/2565.0 )* k3g_double[j] + ( 1859.0/4104.0 )* k4g_double[j] - ( 11.0 / 40.0 )* k5g_double[j];
  }

  tmpw  = w  - (8.0/27.0)*k1w  + 2.0*k2w  - (3544.0/2565.0)*k3w  + (1859.0/4104.0)*k4w  - ( 11.0 / 40.0 )*k5w;

  if ( tmpv > V_peak ) { tmpv = V_peak; }

  k6v = _lintegration_step*RKV(_threadargs_, tmpv, _p_tmpg, tmpw);
  RKDG(_threadargs_, _p_k6dg, _p_tmpdg);
  mul1(_lintegration_step, _p_k6dg);
  RKG(_threadargs_, _p_k6g, _p_tmpdg, _p_tmpg);
  mul1(_lintegration_step, _p_k6g);
  k6w = _lintegration_step*RKW(_threadargs_, tmpv, tmpw);

  tmpv = V_M + (25.0 / 216.0)*k1v + (1408.0/2565.0)*k3v + (2197.0/4104.0)*k4v - (1.0/5.0)*k5v;
  
  double * k6dg_double = vector_vec(_p_k6dg), *k6g_double = vector_vec(_p_k6g);
  for (j=0; j<nb_receptors; j++)
  {
    tmpdg_double[j] = dg_in_double[j] + ( 25.0 / 216.0 )*k1dg_double[j] + (1408.0/2565.0)*k3dg_double[j] + (2197.0/4104.0)*k4dg_double[j] - (1.0/5.0)*k5dg_double[j];
     tmpg_double[j] =  g_in_double[j] + ( 25.0 / 216.0 )* k1g_double[j] + (1408.0/2565.0)* k3g_double[j] + (2197.0/4104.0)* k4g_double[j] - (1.0/5.0)* k5g_double[j];
  }

  tmpw = w + (25.0 / 216.0)*k1w + (1408.0/2565.0)*k3w + (2197.0/4104.0)*k4w - (1.0/5.0)*k5w;


  vs = V_M + (16.0/135.0)*k1v + (6656.0/12825.0)*k3v + (28561.0/56430.0)*k4v - (9.0/50.0)*k5v + (2.0/55.0)*k6v;

  for (j=0; j<nb_receptors; j++)
  {
    dgs_double[j] = dg_in_double[j] + k1dg_double[j]*(16.0/135.0)+k3dg_double[j]*(6656.0/12825.0)+k4dg_double[j]*(28561.0/56430.0)-k5dg_double[j]*(9.0 / 50.0)+k6dg_double[j]*(2.0 / 55.0);
     gs_double[j]  = g_in_double[j] +  k1g_double[j]*(16.0/135.0)+ k3g_double[j]*(6656.0/12825.0)+ k4g_double[j]*(28561.0/56430.0)- k5g_double[j]*(9.0 / 50.0)+ k6g_double[j]*(2.0 / 55.0);
  }

  ws = w + (16.0/135.0)*k1w + (6656.0/12825.0)*k3w + (28561.0/56430.0)*k4w - (9.0/50.0)*k5w + (2.0/55.0)*k6w;

  screwu =  (vs - tmpv)*(vs-tmpv) + (ws - tmpw)*(ws-tmpw);
  for (j=0; j<nb_receptors; j++)
  {
    screwu+= (dgs_double[j]-tmpdg_double[j])*(dgs_double[j]-tmpdg_double[j]) + (gs_double[j] - tmpg_double[j])*(gs_double[j] - tmpg_double[j]);
  }

// printf("screwu: %010.5f\n",screwu);
  R = sqrt(screwu);
  norm = tmpv*tmpv + tmpw*tmpw;

  for (j=0; j<nb_receptors; j++)
  {
    norm+= tmpg_double[j]*tmpg_double[j] + tmpdg_double[j]*tmpdg_double[j];
  }
  norm = sqrt(norm);

// printf("R: %010.5f eps: %010.5f norm: %010.5f --*\n", R, epsilon, norm);
  if (R == R) {
//     printf("I don't think that R is Nan\n")
    if ( R <= epsilon*norm ) {
      flag_update_time = 1;
      V_M  = tmpv;
      if (nb_receptors>0)
      {
        setArray( _p_g_in,  _p_tmpg);
        setArray(_p_dg_in, _p_tmpdg);
      }
      w  = tmpw;
    } else {
      if ( force_integration ) {
        V_M  = tmpv;
        if (nb_receptors>0)
        {
          setArray( _p_g_in,  _p_tmpg);
          setArray(_p_dg_in, _p_tmpdg);
        }
        w  = tmpw;
      } else {
        // printf("Repeating timestep!\n");
      }
    }
  } else {
    printf("There was a problem with R: %010.5f\n", R);
  }

  if ( R==R && R >= 1e-6 ) {
    delta = 0.84 * pow( epsilon/R , 0.2 );
  } else {
    delta = 1.0;
    R = 0.0;
  }

  if (delta <= 1e-9 || delta >= 100. || force_integration ) {
    delta = 1.0;
  }
  ENDVERBATIM
  ADVANCE_RK = integration_step * delta
:  printf("delta: %010.5f R: %010.5f epsilon: %010.5f new step: %020.15f\n", delta, R, epsilon, ADVANCE_RK)
}

BREAKPOINT {
    LOCAL integration_step, new_integration_step
    VERBATIM
    _lintegration_step = fmin(0.01, dt);
    ENDVERBATIM
    WHILE (t0 < t - 1e-6)
    {
      flag_update_time = 0
      VERBATIM
        if (current_stimsize>0 && iterator<vector_capacity(_p_I_stim))
        {
          double* I_stim_double = vector_vec(_p_I_stim);
          loc_i_stim = I_stim_double[(int)iterator];
        }
      ENDVERBATIM
      new_integration_step = ADVANCE_RK(integration_step)
      if ( flag_update_time )
      {
        if (refractory >0)
        {
          V_M = V_reset
        }
        if ( V_M >= V_peak )
        {
          refractory = t_ref / dt + 1
          V_M = V_reset
          w = w + b
        }
        t0 = t0 + integration_step
      }
      if ( new_integration_step < t - t0 ) {
        integration_step = new_integration_step
      } else {
        integration_step = t - t0
      }
    }
    t0 = t
    if (refractory >0)
    {
      spike = 2 : greater than one
      refractory = refractory -1
    }
    else
    {
      spike = 0
    }
  iterator = iterator +1
}

NET_RECEIVE (id_, receptor) {
  INITIAL
  {}
  VERBATIM
   int id_syn = (int)_args[0];
   double* dg_in_double = vector_vec(_p_dg_in),
         *g0_double = vector_vec(_p_g0), 
         *weights_d = vector_vec(_p_weights), 
         *tau_recs_d = vector_vec(_p_tau_recs),
         *tau_facs_d = vector_vec(_p_tau_facs),
         *Us_d = vector_vec(_p_Us),
         *lastspikes_d = vector_vec(_p_t_lastspikes),
         *xs_d = vector_vec(_p_xs),
         *us_d = vector_vec(_p_us);
   double h, x_decay, u_decay;
   h = t - lastspikes_d[id_syn];
   x_decay = exp( -h / tau_recs_d[id_syn] );
   if(tau_facs_d[id_syn] < 1.0e-10)
   {
     u_decay =0.0;
   }
   else
   {
     u_decay = exp( -h / tau_facs_d[id_syn] );
   }
   xs_d[id_syn] = 1. + ( xs_d[id_syn] - xs_d[id_syn] * us_d[id_syn] - 1. ) * x_decay;
   us_d[id_syn] = Us_d[id_syn] + us_d[id_syn] * ( 1. - Us_d[id_syn] ) * u_decay;
   dg_in_double[(int)_args[1]] += g0_double[(int)_args[1]] * xs_d[id_syn] * us_d[id_syn] * weights_d[id_syn];
   lastspikes_d[id_syn] = t;
  ENDVERBATIM
}

PROCEDURE setPostsyn(){
VERBATIM
  int j=0;
  void *vv = vector_arg(1);
  u_int32_t dsize2 = (u_int32_t)vector_capacity(vv);
  _p_E_rev = vector_new1((int)dsize2);
  double *v1_double = vector_vec(_p_E_rev), *v2_double = vector_vec(vv);
  for (j=0; j<dsize2; j++)
  {
    v1_double[j] = v2_double[j];
  }
  nb_receptors = dsize2;
  //printf("%d\n", dsize2);

  vv = vector_arg(2);
  dsize2 = (u_int32_t)vector_capacity(vv);
  _p_tau_rise = vector_new1((int)dsize2);
  v1_double = vector_vec(_p_tau_rise), v2_double = vector_vec(vv);
  for (j=0; j<dsize2; j++)
  {
    v1_double[j] = v2_double[j];
  }

  vv = vector_arg(3);
  dsize2 = (u_int32_t)vector_capacity(vv);
  _p_tau_decay = vector_new1((int)dsize2);
  v1_double = vector_vec(_p_tau_decay), v2_double = vector_vec(vv);
  for (j=0; j<dsize2; j++)
  {
    v1_double[j] = v2_double[j];
  }
  // vv = (void**)(&_p_tau_rise);
  // *vv = (void*)0;
  // if (ifarg(2)) {
  //   *vv = vector_arg(2);
  // }

  // vv = (void**)(&_p_tau_decay);
  // *vv = (void*)0;
  // if (ifarg(3)) {
  //   *vv = vector_arg(3);
  // }
ENDVERBATIM
}

PROCEDURE setI_stim(){
VERBATIM
  int j=0;
  void *vv = vector_arg(1);
  u_int32_t dsize2 = (u_int32_t)vector_capacity(vv);
  //printf("%d\n",dsize2);
  if (current_stimsize>0 && dsize2>current_stimsize)
  {
    vector_resize(_p_I_stim, dsize2);
    current_stimsize = dsize2;
  }
  else
  {
    _p_I_stim = vector_new1((int)dsize2);
    current_stimsize = dsize2;
    double *v1_double = vector_vec(_p_I_stim);
    for (j=0; j<dsize2; j++)
      v1_double[j] = 0.0;
  }
  double *v1_double = vector_vec(_p_I_stim), *v2_double = vector_vec(vv);
  for (j=0; j<dsize2; j++)
  {
    v1_double[j] += v2_double[j];
  }
ENDVERBATIM
}

FUNCTION initSynapse(syn_size){
VERBATIM
  if (nb_synapses ==0){
    _p_weights = vector_new1(_lsyn_size);
    _p_tau_recs = vector_new1(_lsyn_size);
    _p_tau_facs = vector_new1(_lsyn_size);
    _p_Us = vector_new1(_lsyn_size);
    _p_t_lastspikes = vector_new1(_lsyn_size);
    _p_xs = vector_new1(_lsyn_size);
    _p_us = vector_new1(_lsyn_size);
  }
  else
  {
    vector_resize(_p_weights, nb_synapses+_lsyn_size);
    vector_resize(_p_tau_recs, nb_synapses+_lsyn_size);
    vector_resize(_p_tau_facs, nb_synapses+_lsyn_size);
    vector_resize(_p_Us, nb_synapses+_lsyn_size);
    vector_resize(_p_t_lastspikes, nb_synapses+_lsyn_size);
    vector_resize(_p_xs, nb_synapses+_lsyn_size);
    vector_resize(_p_us, nb_synapses+_lsyn_size);
  }
ENDVERBATIM
  initSynapse = nb_synapses
  nb_synapses = nb_synapses + syn_size
}

PROCEDURE addSynapse(id, weight, tau_rec, tau_fac, U, t_lastspike, x, u){
VERBATIM
  double *weights_d = vector_vec(_p_weights), 
         *tau_recs_d = vector_vec(_p_tau_recs),
         *tau_facs_d = vector_vec(_p_tau_facs),
         *Us_d = vector_vec(_p_Us),
         *lastspikes_d = vector_vec(_p_t_lastspikes),
         *xs_d = vector_vec(_p_xs),
         *us_d = vector_vec(_p_us);

  weights_d[(int)_lid] = _lweight;
  tau_recs_d[(int)_lid] = _ltau_rec;
  tau_facs_d[(int)_lid] = _ltau_fac;
  Us_d[(int)_lid] = _lU;
  lastspikes_d[(int)_lid] = _lt_lastspike;
  xs_d[(int)_lid] = _lx;
  us_d[(int)_lid] = _lu;
ENDVERBATIM
}
