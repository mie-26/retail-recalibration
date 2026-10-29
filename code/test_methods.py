import unittest
import numpy as np
from feedback_variants import multi,eta_path,replay,ordered

class CausalTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261025);T,N=80,12
        self.z=dict(truth=rng.poisson(4,(T,N)).astype(float),base=rng.uniform(-2,9,(T,N,2)),origins=np.arange(100,100+T),scale=np.linspace(1,3,N),groups=np.arange(N)%3,versions=np.repeat(np.arange(4),20))
        self.z['truth'][::4,0]=np.nan
    def test_exact_default(self):
        z=self.z;eta=eta_path(z,7);a,_=multi(z,7,eta)
        b,_,_=replay(z['truth'],z['base'],z['scale'],z['groups'],z['origins'],z['versions'],7,'R5_multiqt_nonnegative')
        np.testing.assert_array_equal(a,b)
    def test_future_labels(self):
        z=self.z;zz={k:v.copy() for k,v in z.items()};zz['truth'][34:]=999
        for rate in [.03,.1,.3]:
            a,_=multi(z,7,eta_path(z,7),rate);b,_=multi(zz,7,eta_path(zz,7),rate)
            np.testing.assert_array_equal(a[:41],b[:41])
    def test_shadow_clamp_nonincrease(self):
        z=self.z;a,_=multi(z,7,eta_path(z,7),played=False,clamp=False);b=np.maximum(a,0)
        from feedback_variants import TAUS
        ea=z['truth'][:,:,None]-a;eb=z['truth'][:,:,None]-b;valid=np.isfinite(z['truth'])
        self.assertTrue(np.all((np.maximum(eb*TAUS,eb*(TAUS-1))-np.maximum(ea*TAUS,ea*(TAUS-1)))[valid]<=1e-12))
    def test_current_version_maturity(self):
        z=self.z;q,logs,_=replay(z['truth'],z['base'],z['scale'],z['groups'],z['origins'],z['versions'],7,'R3_isolate')
        for start in [0,20,40,60]:
            np.testing.assert_array_equal(q[start:start+7],ordered(z['base'][start:start+7]))
        self.assertTrue(all(r['latest_maturity']<=r['origin'] for r in logs))
    def test_missing_never_zero_filled(self):
        z=self.z;z['truth'][:]=np.nan;q,_=multi(z,7,eta_path(z,7))
        np.testing.assert_allclose(q,ordered(z['base']),rtol=1e-15,atol=1e-15)

if __name__=='__main__':unittest.main()
