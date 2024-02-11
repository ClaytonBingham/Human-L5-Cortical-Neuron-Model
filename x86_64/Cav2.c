/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__Cav2
#define _nrn_initial _nrn_initial__Cav2
#define nrn_cur _nrn_cur__Cav2
#define _nrn_current _nrn_current__Cav2
#define nrn_jacob _nrn_jacob__Cav2
#define nrn_state _nrn_state__Cav2
#define _net_receive _net_receive__Cav2 
#define rates rates__Cav2 
#define states states__Cav2 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define pbar _p[0]
#define ica _p[1]
#define i _p[2]
#define igate _p[3]
#define nc _p[4]
#define m _p[5]
#define cai _p[6]
#define cao _p[7]
#define T _p[8]
#define E _p[9]
#define zeta _p[10]
#define qt _p[11]
#define Dm _p[12]
#define _g _p[13]
#define _ion_cai	*_ppvar[0]._pval
#define _ion_cao	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_ghk(void);
 static void _hoc_kelvinfkt(void);
 static void _hoc_mgateFlip(void);
 static void _hoc_rates(void);
 static void _hoc_taumfkt(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_Cav2", _hoc_setdata,
 "ghk_Cav2", _hoc_ghk,
 "kelvinfkt_Cav2", _hoc_kelvinfkt,
 "mgateFlip_Cav2", _hoc_mgateFlip,
 "rates_Cav2", _hoc_rates,
 "taumfkt_Cav2", _hoc_taumfkt,
 0, 0
};
#define ghk ghk_Cav2
#define kelvinfkt kelvinfkt_Cav2
#define mgateFlip mgateFlip_Cav2
#define taumfkt taumfkt_Cav2
 extern double ghk( double , double , double , double );
 extern double kelvinfkt( double );
 extern double mgateFlip( );
 extern double taumfkt( double );
 /* declare global and static user variables */
#define gateCurrent gateCurrent_Cav2
 double gateCurrent = 0;
#define monovalPerm monovalPerm_Cav2
 double monovalPerm = 0;
#define monovalConc monovalConc_Cav2
 double monovalConc = 140;
#define minf minf_Cav2
 double minf = 0;
#define punit punit_Cav2
 double punit = 3.29e-13;
#define taum taum_Cav2
 double taum = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gateCurrent_Cav2", "1",
 "punit_Cav2", "cm3/s",
 "monovalConc_Cav2", "mM",
 "monovalPerm_Cav2", "1",
 "minf_Cav2", "1",
 "taum_Cav2", "ms",
 "pbar_Cav2", "cm/s",
 "ica_Cav2", "mA/cm2",
 "i_Cav2", "mA/cm2",
 "igate_Cav2", "mA/cm2",
 "nc_Cav2", "1/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double m0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "gateCurrent_Cav2", &gateCurrent_Cav2,
 "punit_Cav2", &punit_Cav2,
 "monovalConc_Cav2", &monovalConc_Cav2,
 "monovalPerm_Cav2", &monovalPerm_Cav2,
 "minf_Cav2", &minf_Cav2,
 "taum_Cav2", &taum_Cav2,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Cav2",
 "pbar_Cav2",
 0,
 "ica_Cav2",
 "i_Cav2",
 "igate_Cav2",
 "nc_Cav2",
 0,
 "m_Cav2",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 14, _prop);
 	/*initialize range parameters*/
 	pbar = 3e-05;
 	_prop->param = _p;
 	_prop->param_size = 14;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Cav2_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 14, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Cav2 /home/clayton/Desktop/Projects/CorticospinalTractModeling/2_FitBiophysics/L5Cell/Cav2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double e0 = 1.60217646e-19;
 static double q10 = 2.7;
 static double F = 9.6485e4;
 static double R = 8.3145;
 static double cv = 19;
 static double ck = 5.5;
 static double zm = 4.6244;
static int _reset;
static char *modelname = "Cav2.1 (P-type) calcium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / taum ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taum )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taum)))*(- ( ( ( minf ) ) / taum ) / ( ( ( ( - 1.0 ) ) ) / taum ) - m) ;
   }
  return 0;
}
 
double ghk (  double _lv , double _lci , double _lco , double _lz ) {
   double _lghk;
 E = ( 1e-3 ) * _lv ;
   zeta = ( _lz * F * E ) / ( R * T ) ;
   if ( fabs ( 1.0 - exp ( - zeta ) ) < 1e-6 ) {
     _lghk = ( 1e-6 ) * ( _lz * F ) * ( _lci - _lco * exp ( - zeta ) ) * ( 1.0 + zeta / 2.0 ) ;
     }
   else {
     _lghk = ( 1e-6 ) * ( _lz * zeta * F ) * ( _lci - _lco * exp ( - zeta ) ) / ( 1.0 - exp ( - zeta ) ) ;
     }
   
return _lghk;
 }
 
static void _hoc_ghk(void) {
  double _r;
   _r =  ghk (  *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 hoc_retpushx(_r);
}
 
static int  rates (  double _lv ) {
   minf = 1.0 / ( 1.0 + exp ( - ( _lv + cv ) / ck ) ) ;
   taum = ( 1e3 ) * taumfkt ( _threadargscomma_ _lv ) / qt ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double taumfkt (  double _lv ) {
   double _ltaumfkt;
  if ( _lv > - 50.0 ) {
     _ltaumfkt = 0.000191 + 0.00376 * exp ( - pow( ( ( _lv + 41.9 ) / 27.8 ) , 2.0 ) ) ;
     }
   else {
     _ltaumfkt = 0.00026367 + 0.1278 * exp ( 0.10327 * _lv ) ;
     }
    
return _ltaumfkt;
 }
 
static void _hoc_taumfkt(void) {
  double _r;
   _r =  taumfkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double kelvinfkt (  double _lt ) {
   double _lkelvinfkt;
  _lkelvinfkt = 273.19 + _lt ;
    
return _lkelvinfkt;
 }
 
static void _hoc_kelvinfkt(void) {
  double _r;
   _r =  kelvinfkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double mgateFlip (  ) {
   double _lmgateFlip;
 _lmgateFlip = ( minf - m ) / taum ;
   
return _lmgateFlip;
 }
 
static void _hoc_mgateFlip(void) {
  double _r;
   _r =  mgateFlip (  );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  m = m0;
 {
   nc = pbar / punit ;
   qt = pow( q10 , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   T = kelvinfkt ( _threadargscomma_ celsius ) ;
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  cai = _ion_cai;
  cao = _ion_cao;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   ica = ( 1e3 ) * pbar * m * ghk ( _threadargscomma_ v , cai , cao , 2.0 ) ;
   igate = nc * ( 1e6 ) * e0 * zm * mgateFlip ( _threadargs_ ) ;
   if ( gateCurrent  != 0.0 ) {
     i = igate ;
     }
   }
 _current += ica;
 _current += i;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  cai = _ion_cai;
  cao = _ion_cao;
 _g = _nrn_current(_v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  cai = _ion_cai;
  cao = _ion_cao;
 { error =  states();
 if(error){fprintf(stderr,"at line 98 in file Cav2.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(m) - _p;  _dlist1[0] = &(Dm) - _p;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/home/clayton/Desktop/Projects/CorticospinalTractModeling/2_FitBiophysics/L5Cell/Cav2.mod";
static const char* nmodl_file_text = 
  "TITLE Cav2.1 (P-type) calcium channel\n"
  "\n"
  "COMMENT\n"
  "\n"
  "NEURON implementation of a Cav2.1 calcium channel\n"
  "Kinetical scheme: Hodgkin-Huxley (m), no inactivation\n"
  "\n"
  "Model includes a calculation of the gating current\n"
  "\n"
  "Modified from Khaliq et al., J. Neurosci. 23(2003)4899\n"
  "\n"
  "Reference: Akemann et al. (2009) 96:3959-3976\n"
  "\n"
  "Laboratory for Neuronal Circuit Dynamics\n"
  "RIKEN Brain Science Institute, Wako City, Japan\n"
  "http://www.neurodynamics.brain.riken.jp\n"
  "\n"
  "Date of Implementation: April 2007\n"
  "Contact: akemann@brain.riken.jp\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX Cav2\n"
  "	USEION ca READ cai, cao WRITE ica\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	RANGE pbar, ica, i, igate, nc\n"
  "	GLOBAL minf, taum\n"
  "	GLOBAL gateCurrent, punit \n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "	(nA) = (nanoamp)\n"
  "	(pA) = (picoamp)\n"
  "	(S)  = (siemens)\n"
  "	(nS) = (nanosiemens)\n"
  "	(pS) = (picosiemens)\n"
  "	(um) = (micron)\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)		\n"
  "}\n"
  "\n"
  "CONSTANT {\n"
  "	e0 = 1.60217646e-19 (coulombs)\n"
  "	q10 = 2.7\n"
  "	F = 9.6485e4 (coulombs)\n"
  "	R = 8.3145 (joule/kelvin)\n"
  "\n"
  "	cv = 19 (mV)\n"
  "	ck = 5.5 (mV)\n"
  "\n"
  "	zm = 4.6244 (1)				: gating charge\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	gateCurrent = 0 (1)			: gating currents ON = 1 OFF = 0\n"
  "	\n"
  "	pbar = 3e-5 (cm/s)\n"
  "	punit = 3.290e-13 (cm3/s)		: unitary calcium permeability\n"
  "	\n"
  "	monovalConc = 140 (mM)\n"
  "	monovalPerm = 0 (1)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	celsius (degC)\n"
  "	v (mV)\n"
  "	cai (mM)\n"
  "	cao (mM)\n"
  "\n"
  "	ica (mA/cm2)\n"
  "	i (mA/cm2)\n"
  "	igate (mA/cm2)\n"
  "	\n"
  "	nc (1/cm2)				: membrane density of channel\n"
  "\n"
  "      minf (1) \n"
  "	taum (ms)\n"
  "	T (kelvin)\n"
  "	E (volt)\n"
  "	zeta (1)\n"
  "	qt (1)\n"
  "}\n"
  "\n"
  "STATE { m }\n"
  "\n"
  "INITIAL {\n"
  "	nc = pbar / punit\n"
  "	qt = q10^((celsius-22 (degC))/10 (degC))\n"
  "	T = kelvinfkt( celsius )\n"
  "	rates(v)\n"
  "	m = minf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	ica = (1e3) * pbar * m * ghk(v, cai, cao, 2)\n"
  "	igate = nc * (1e6) * e0 * zm * mgateFlip()\n"
  "\n"
  "	if (gateCurrent != 0) { \n"
  "		i = igate\n"
  "	}\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "	rates(v)\n"
  "	m' = (minf-m)/taum\n"
  "}\n"
  "\n"
  "FUNCTION ghk( v (mV), ci (mM), co (mM), z )  (coulombs/cm3) { \n"
  "	E = (1e-3) * v\n"
  "      zeta = (z*F*E)/(R*T)	\n"
  "	\n"
  "	: ci = ci + (monovalPerm) * (monovalConc) :Monovalent permeability\n"
  "\n"
  "	if ( fabs(1-exp(-zeta)) < 1e-6 ) {\n"
  "	ghk = (1e-6) * (z*F) * (ci - co*exp(-zeta)) * (1 + zeta/2)\n"
  "	} else {\n"
  "	ghk = (1e-6) * (z*zeta*F) * (ci - co*exp(-zeta)) / (1-exp(-zeta))\n"
  "	}\n"
  "}\n"
  "\n"
  "PROCEDURE rates( v (mV) ) {\n"
  "	minf = 1 / ( 1 + exp(-(v+cv)/ck) )\n"
  "	taum = (1e3) * taumfkt(v)/qt\n"
  "}\n"
  "\n"
  "FUNCTION taumfkt( v (mV) ) (s) {\n"
  "	UNITSOFF\n"
  "	if ( v > -50 ) {\n"
  "	taumfkt = 0.000191 + 0.00376 * exp(-((v+41.9)/27.8)^2)\n"
  "	} else {\n"
  "	taumfkt = 0.00026367 + 0.1278 * exp(0.10327*v)\n"
  "	}\n"
  "	UNITSON\n"
  "}\n"
  "\n"
  "FUNCTION kelvinfkt( t (degC) )  (kelvin) {\n"
  "	UNITSOFF\n"
  "	kelvinfkt = 273.19 + t\n"
  "	UNITSON\n"
  "}\n"
  "\n"
  "FUNCTION mgateFlip() (1/ms) {\n"
  "	mgateFlip = (minf-m)/taum\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
