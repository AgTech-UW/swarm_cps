"""The AI Foreman: a language model that supervises the crew without steering it.

It chooses what an operator or a hand-written program would choose anyway,
the formation, the leader's speed and the sideways offset of the working
passes, from the operator's instruction in plain words. It never outputs a
motion command. Everything it asks for goes through SwarmController.request():
read leniently, checked strictly, and passed through the safety filter.

The prompt, the observation and the reply format follow shepherd's
farmsupervisor.py (prompt version 3) so that the simulator study and the
rovers speak the same protocol; lengths here are metres on the lab floor.

Clients use only the standard library. OllamaClient talks to a local Ollama
server (the paper's set-up: qwen2.5:7b, temperature 0.3, JSON mode).
OpenAIClient talks to any OpenAI-compatible /v1/chat/completions endpoint
(llama.cpp server, LM Studio, vLLM). The host is a setting, so the model can
run on the perception laptop, on another machine, or on a rover's own
Raspberry Pi.
"""
import json
import time
import urllib.request

from . import formation as F

PROMPT_VERSION = 'rover-1'

SYSTEM = """You supervise a team of {N} autonomous machines: one LEADER and {n} FOLLOWERS. They are small rovers working a scale model of a farm field laid out on a floor; treat it as the real field. An autopilot steers the leader along the planned route and stops it at the end. You decide three things: the FORMATION the followers hold, the leader's SPEED, and the sideways OFFSET of the working passes.

TODAY'S INSTRUCTION FROM THE OPERATOR:
"{instruction}"

THE FIELD (x = {px}, y = {py}, metres). The floor covers x {x0:g}-{x1:g}, y {y0:g}-{y1:g}. Route: {route}. {walls}{zones}Hazards are reported when they come within {sr:g} m: a rock is solid. Nothing may touch a wall, a rock or another machine.

THE MACHINES. Leader speed {smin:g}-{smax:g} m/s. Followers can go up to {vmax:g} m/s and turn no tighter than {R:g} m. Followers do not plan around obstacles; each drives along a line through its place in the formation, like autosteer, and speeds up or slows down to keep its place. As a last resort a machine about to run into a rock swerves and slows to miss it, then returns to its line; this reflex does not cover walls, so plan as if it were not there. Forming up after a formation change takes several seconds, and in single file they cannot pass one another.

FORMATIONS. A follower's "lane" is its sideways position relative to the leader (+ = left of travel). Ground is worked where a machine passes, so wide formations get more done per pass and single file works one {sw:g} m strip.
{table}
You may give a custom formation instead: {{"shape": "custom", "slots": [[along, lane], ... {n} pairs]}}, along < 0 behind the leader, with every pair of machines (the leader included) more than {gap:g} m apart.

OFFSET moves the working passes, and the turn between them, sideways as one piece: {offPlus} (+) or {offMinus} (-), in metres, whichever way the machines are travelling, within the range the observation gives for each formation. It has no effect before the first working pass, the line takes about a metre of travel to move over, and it stays where you put it until you change it, so set it for the passes still to come and not only for the one you are on. Lanes and the lineLeft/lineRight formations are relative to the direction of travel; observation.pass.leftOfTravelIs says which side is left on the current pass.

observation.formationsThatFit says, for each formation, whether it can be taken up now: every lane stays on the floor and clear of the walls from the tail of the crew to {reach:g} m ahead, no lane has to turn tighter than a machine can, and every follower can drive to its new place without crossing a wall. A formation that does not fit is not taken up; it is held and taken up when it fits. Speed does not change how wide a formation is.

You are called at the start, at each new phase, when a hazard comes into range, when the formations that fit change, and otherwise every {period:g} s. Your choice stays in force until you change it.

Reply with ONE JSON object:
  {{"thought": "<one or two sentences>", "formation": "<name>" or {{custom}}, "speed": <m/s>, "offset": <m>}}
{lessons}"""


def build_system(ctl, instruction, lessons=None):
    ar, p, r = ctl.arena, ctl.p, ctl.route
    plus, minus = ar.offset_words(r)
    legs = []
    for lg in r.legs:
        x, y, th, _ = r.at((lg['s0'] + lg['s1']) / 2)
        if lg['work']:
            xa, ya, _, _ = r.at(lg['s0'] + 0.01)
            xb, yb, _, _ = r.at(lg['s1'] - 0.01)
            line = ('y = %.2f' % ya) if abs(ya - yb) < 0.05 else ('x = %.2f' % xa)
            legs.append('%s (working pass heading %s along %s)' % (lg['name'], ar.compass(th), line))
        else:
            legs.append(lg['name'])
    walls = ''
    if ar.walls:
        xs = sorted(set(round(w['a'][0], 2) for w in ar.walls if abs(w['a'][0] - w['b'][0]) < 1e-6))
        ys = sorted(set(round(w['a'][1], 2) for w in ar.walls if abs(w['a'][1] - w['b'][1]) < 1e-6))
        where = ' and '.join(['x = %g' % v for v in xs] + ['y = %g' % v for v in ys]) or 'the floor'
        walls = 'A wall along %s separates the yard from the field, with one gate in it; the route goes through the gate. ' % where
    zones = ''
    if ar.zones:
        zones = 'Named areas of the floor: ' + '; '.join(
            '"%s" covers x %g-%g, y %g-%g (%s)' % (z['name'], z['x0'], z['x1'], z['y0'], z['y1'],
                                                    ar.describe_position((z['x0'] + z['x1']) / 2, (z['y0'] + z['y1']) / 2))
            for z in ar.zones) + '. '
    les = ''
    if lessons:
        les = '\nLESSONS FROM EARLIER RUNS:\n' + '\n'.join('- ' + str(x) for x in lessons) + '\n'
    return SYSTEM.format(
        N=ctl.n + 1, n=ctl.n, instruction=instruction or 'Work the field.', px=ar.plus_x, py=ar.plus_y,
        x0=ar.x0, x1=ar.x1, y0=ar.y0, y1=ar.y1, route=' -> '.join(legs), walls=walls, zones=zones,
        sr=p.sensor_range, smin=p.speed_range[0], smax=p.speed_range[1], vmax=p.v_max, R=p.min_turn_radius,
        sw=ar.swath, table=F.table(ctl.n, p.spacing, ar.swath),
        gap=max(p.protected, 2 * p.body_radius + 0.05), offPlus=plus, offMinus=minus, reach=p.fit_reach,
        period=p.period, lessons=les)


def parse_json(text):
    """A model reply -> dict, or None. Tolerates text around the object and
    code fences."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    a, b = text.find('{'), text.rfind('}')
    if a >= 0 and b > a:
        try:
            v = json.loads(text[a:b + 1])
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


class OllamaClient:
    def __init__(self, model='qwen2.5:7b', host='http://127.0.0.1:11434', temperature=0.3, timeout=120, num_ctx=8192):
        self.model, self.host = model, host.rstrip('/')
        self.temperature, self.timeout, self.num_ctx = temperature, timeout, num_ctx

    def chat(self, messages):
        body = dict(model=self.model, messages=messages, stream=False, format='json',
                    options=dict(temperature=self.temperature, num_ctx=self.num_ctx))
        req = urllib.request.Request(self.host + '/api/chat', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())['message']['content']

    def ping(self):
        """(ok, message): is the server up and is the model pulled?"""
        try:
            with urllib.request.urlopen(self.host + '/api/tags', timeout=5) as r:
                names = [m.get('name', '') for m in json.loads(r.read()).get('models', [])]
        except Exception as e:
            return False, 'no Ollama server at %s (%s). Start it with: ollama serve' % (self.host, e)
        if not any(n == self.model or n.startswith(self.model + ':') or n.split(':')[0] == self.model for n in names):
            return False, 'model %s is not pulled (have: %s). Run: ollama pull %s' % (self.model, ', '.join(names) or 'none', self.model)
        return True, 'Ollama at %s has %s' % (self.host, self.model)


class OpenAIClient:
    def __init__(self, model, host='http://127.0.0.1:8080', temperature=0.3, timeout=120, api_key=None):
        self.model, self.host = model, host.rstrip('/')
        self.temperature, self.timeout, self.api_key = temperature, timeout, api_key

    def chat(self, messages):
        body = dict(model=self.model, messages=messages, temperature=self.temperature,
                    response_format=dict(type='json_object'))
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        url = self.host + ('/chat/completions' if self.host.endswith('/v1') else '/v1/chat/completions')
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())['choices'][0]['message']['content']

    def ping(self):
        try:
            self.chat([dict(role='user', content='Reply with the JSON object {"ok": true}')])
            return True, 'endpoint %s answers' % self.host
        except Exception as e:
            return False, 'no answer from %s (%s)' % (self.host, e)


class Foreman:
    """The decider. decide(observation) returns the model's choice as a dict,
    or None if the reply could not be read (the current choice then stays in
    force). Keeps a transcript of everything it was shown and everything it
    said."""
    name = 'llm'

    def __init__(self, client, history_turns=2, votes=1, lessons=None, verbose=False):
        self.client = client
        self.history_turns = history_turns
        self.votes = max(1, int(votes))
        self.lessons = lessons
        self.verbose = verbose
        self.system = ''
        self.history = []
        self.transcript = []
        self.n_calls = self.n_bad = 0
        self.last_thought = ''

    def begin(self, ctl, instruction=''):
        self.set_system(build_system(ctl, instruction, self.lessons))

    def set_system(self, text):
        self.system = text
        self.history = []

    def _ask(self, msgs):
        t0 = time.time()
        try:
            text = self.client.chat(msgs)
        except Exception as e:
            text = json.dumps(dict(error=str(e)))
        self.n_calls += 1
        return text, time.time() - t0

    def decide(self, ob):
        msg = dict(role='user', content=json.dumps(ob))
        msgs = [dict(role='system', content=self.system)] + self.history[-2 * self.history_turns:] + [msg]
        replies = []
        for _ in range(self.votes):
            text, sec = self._ask(msgs)
            replies.append((text, sec, parse_json(text)))
        good = [(t, s, r) for t, s, r in replies if isinstance(r, dict) and 'formation' in r]
        rec = dict(observation=ob, replies=[t for t, _, _ in replies], seconds=round(sum(s for _, s, _ in replies), 2))
        if not good:
            self.n_bad += 1
            rec['choice'] = None
            self.transcript.append(rec)
            self.history += [msg, dict(role='assistant', content=replies[0][0])]
            return None
        # with several votes, take the most common formation (self-consistency)
        keys = [json.dumps(r.get('formation'), sort_keys=True) for _, _, r in good]
        top = max(set(keys), key=keys.count)
        text, _, rep = good[keys.index(top)]
        self.history += [msg, dict(role='assistant', content=text)]
        self.last_thought = str(rep.get('thought', ''))[:300]
        rec['choice'] = dict(formation=rep.get('formation'), speed=rep.get('speed'), offset=rep.get('offset'))
        rec['thought'] = self.last_thought
        self.transcript.append(rec)
        if self.verbose:
            print('  foreman: %s | %s' % (rec['choice'], self.last_thought))
        return rep


class MockModel:
    """Offline stand-in for a model, for rehearsing the plumbing with no GPU.
    It is a few lines of keyword matching, not a language model: it knows
    that "north" / "south" / "east" / "west" in an instruction name a side to
    keep away from, that "slow" and "fast" are about speed, and that narrow
    places want single file. Every ninth reply is malformed on purpose, to
    exercise the path a real model's bad replies take."""

    def __init__(self, malformed_every=9):
        self.n = 0
        self.malformed_every = malformed_every

    def chat(self, messages):
        self.n += 1
        system = messages[0]['content']
        try:
            ob = json.loads(messages[-1]['content'])
        except Exception:
            ob = {}
        if self.malformed_every and self.n % self.malformed_every == 0:
            return 'Let me think about the best formation here.'
        a = system.find('INSTRUCTION FROM THE OPERATOR:')
        ins = system[a:system.find('THE FIELD', a)].lower() if a >= 0 else ''
        speed = 0.13
        if any(w in ins for w in ('slow', 'careful', 'gently', 'people')):
            speed = 0.08
        if any(w in ins for w in ('fast', 'hurry', 'quick', 'rain')):
            speed = 0.18
        ps = ob.get('pass')
        fits = ob.get('formationsThatFit', {})
        if ps is None:
            return json.dumps(dict(thought='narrow ground, single file', formation='column', speed=speed, offset=0))
        words = ps['offsetMeaning'].replace(',', '').split()
        plus, minus = words[5], words[-1]
        avoid = None
        for side in (plus, minus):
            if side in ins:
                avoid = side
        form, off = 'line', 0.0
        if avoid is not None:
            left_is = ps['leftOfTravelIs']
            form = 'lineRight' if avoid in left_is else 'lineLeft'
            lo, hi = ps['allowedOffsetM'].get(form, [0.0, 0.0])
            off = max(lo, -0.5) if avoid == plus else min(hi, 0.5)
            thought = 'keep away from the %s side: fold to the far side and move the pass over' % avoid
        else:
            thought = 'open ground, work wide'
        if not ps.get('working') or not fits.get(form, True) or ps['metresLeft'] < 0.5:
            if avoid is not None:
                lo, hi = ps['allowedOffsetM'].get('column', [0.0, 0.0])
                off = max(lo, -0.5) if avoid == plus else min(hi, 0.5)
            form = 'column'
        if self.n % 5 == 0:                                   # a preset written the way some models write it
            return json.dumps(dict(thought=thought, formation=dict(shape=form.upper()), speed=speed, offset=off))
        return json.dumps(dict(thought=thought, formation=form, speed=speed, offset=off))

    def ping(self):
        return True, 'mock model (keyword matching, no language model)'
