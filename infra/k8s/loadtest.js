// k6 load test for the claim triage API in kind (make loadtest).
//
// Open model: requests arrive at a fixed rate whether or not the API keeps
// up, like real traffic (a closed loop of busy users hides slowness by
// slowing down with it). Through the gateway (localhost:8090), which
// balances per request. Half the requests send a full claim with an
// adjuster note (both models run), half send only a policy_id (features
// from the online store, triage model only).
//
// Thresholds are the service-level objectives in the README; k6 exits
// non-zero if one fails.
import http from 'k6/http';
import { check } from 'k6';
import { Counter } from 'k6/metrics';

const BASE = __ENV.BASE_URL || 'http://localhost:8090';
const POLICIES = JSON.parse(open(__ENV.POLICIES || '../../.kind/policies.json'));
const CLAIM = open('smoke_claim.json');
const NOTES = [
  'Rear-ended at a red light, bumper and boot damaged.',
  'Windscreen cracked by a stone on the motorway.',
  'Car broken into overnight, radio and bag stolen.',
  'Hail storm dented the roof and bonnet.',
  'Driver taken to hospital with neck pain after a collision.',
];
const PARAMS = { headers: { 'content-type': 'application/json' }, timeout: '10s' };
const byTrack = new Counter('requests_by_track');

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-arrival-rate',
      startRate: 2,
      timeUnit: '1s',
      preAllocatedVUs: 20,
      maxVUs: 100,
      stages: [
        { target: 10, duration: '30s' },
        { target: 10, duration: '60s' },
        { target: 20, duration: '30s' },
        { target: 20, duration: '60s' },
        { target: 30, duration: '30s' },
        { target: 30, duration: '90s' },
        { target: 0, duration: '20s' },
      ],
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    'http_req_duration{endpoint:policy}': ['p(95)<100', 'p(99)<250'],
    'http_req_duration{endpoint:claim}': ['p(95)<300', 'p(99)<500'],
    dropped_iterations: ['count<10'],
  },
};

export default function () {
  let res;
  if (Math.random() < 0.5) {
    const id = POLICIES[Math.floor(Math.random() * POLICIES.length)];
    res = http.post(`${BASE}/predict/policy`, JSON.stringify({ policy_id: id }), {
      ...PARAMS,
      tags: { endpoint: 'policy' },
    });
  } else {
    const claim = JSON.parse(CLAIM);
    claim.adjuster_note = NOTES[Math.floor(Math.random() * NOTES.length)];
    claim.driver_age = 18 + Math.floor(Math.random() * 70);
    res = http.post(`${BASE}/predict`, JSON.stringify(claim), { ...PARAMS, tags: { endpoint: 'claim' } });
  }
  check(res, { 'status 200': (r) => r.status === 200 });
  byTrack.add(1, { track: res.headers['X-Serving-Track'] || 'none' });
}
