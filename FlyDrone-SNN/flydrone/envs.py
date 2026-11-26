"""Batched gym-pybullet-drones environment with a PID teacher.

Uses HoverAviary (single Crazyflie CF2X, target hover position) and the
DSLPIDControl controller from gym-pybullet-drones as a teacher for
supervised (imitation) e-prop.

Observation fed to the network (12 dims, scaled to ~[-1, 1]):
    position error to target (3), roll/pitch/yaw (3), linear velocity (3),
    angular velocity (3)
Actions (normalised in [-1, 1]):
    'rpm'       : 4 motor commands, rpm = HOVER_RPM * (1 + rpm_scale * a)
    'one_d_rpm' : 1 collective command applied to all motors
"""
from __future__ import annotations

import numpy as np

OBS_SCALE = np.array([1, 1, 1, np.pi / 4, np.pi / 4, np.pi, 2, 2, 2, 2 * np.pi, 2 * np.pi, 2 * np.pi],
                     dtype=np.float32)


def encode_obs(obs: np.ndarray) -> np.ndarray:
    """(B, 12) -> (C=25, B): bias + positive / negative half-wave channels.

    Splitting signs keeps input channels non-negative (like firing rates)."""
    o = np.clip(obs / OBS_SCALE, -3, 3)
    x = np.concatenate([np.ones((o.shape[0], 1)), np.maximum(o, 0), np.maximum(-o, 0)], axis=1)
    return x.T.astype(np.float32)


N_INPUTS = 25


class DroneBatch:
    def __init__(self, n_envs=8, act='rpm', ctrl_freq=50, pyb_freq=250, rpm_scale=0.3,
                 init_pos_noise=0.3, init_rpy_noise=0.25, episode_sec=8, seed=0, gui=False):
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

        scale = rpm_scale

        class _Aviary(HoverAviary):
            EPISODE_LEN_SEC = episode_sec

            def _preprocessAction(self, action):
                a = np.clip(np.asarray(action, dtype=np.float64).reshape(1, -1), -1, 1)
                if a.shape[1] == 1:
                    a = np.repeat(a, 4, axis=1)
                return self.HOVER_RPM * (1 + scale * a)

            def _computeTerminated(self):
                return False if self.GUI else super()._computeTerminated()

            def _computeTruncated(self):
                return False if self.GUI else super()._computeTruncated()

        self.act = act
        self.n_actions = 4 if act == 'rpm' else 1
        self.rpm_scale = rpm_scale
        self.rng = np.random.default_rng(seed)
        self.init_pos_noise, self.init_rpy_noise = init_pos_noise, init_rpy_noise
        a_type = ActionType.RPM if act == 'rpm' else ActionType.ONE_D_RPM
        self.envs = [_Aviary(drone_model=DroneModel.CF2X, physics=Physics.PYB, pyb_freq=pyb_freq,
                             ctrl_freq=ctrl_freq, gui=(gui and k == 0), record=False,
                             obs=ObservationType.KIN, act=a_type) for k in range(n_envs)]
        self.ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(n_envs)]
        self.target = np.asarray(self.envs[0].TARGET_POS, dtype=np.float64)
        self.ctrl_dt = 1.0 / ctrl_freq
        self.n = n_envs
        self.t = np.zeros(n_envs, int)
        self.finished_lengths = []
        self.max_steps = int(episode_sec * ctrl_freq)

    # -- helpers -------------------------------------------------------------
    def _state(self, i):
        return self.envs[i]._getDroneStateVector(0)

    def _obs(self, i):
        s = self._state(i)
        pos, rpy, vel, angv = s[0:3], s[7:10], s[10:13], s[13:16]
        return np.concatenate([pos - self.target, rpy, vel, angv]).astype(np.float32)

    def _reset_one(self, i):
        e = self.envs[i]
        xy = self.target[:2] + self.rng.uniform(-0.1, 0.1, 2)
        z = self.rng.uniform(0.05, 0.15)
        e.INIT_XYZS = np.array([[xy[0], xy[1], z]], dtype=np.float64)
        e.INIT_RPYS = np.zeros((1, 3))
        e.reset()
        self.ctrls[i].reset()
        self.t[i] = 0

    # -- API -----------------------------------------------------------------
    def reset(self):
        for i in range(self.n):
            self._reset_one(i)
        return self.observe()

    def observe(self):
        return np.stack([self._obs(i) for i in range(self.n)])

    def teacher(self):
        """PID teacher action, normalised to the action space. (B, A)"""
        out = []
        for i in range(self.n):
            rpm, _, _ = self.ctrls[i].computeControlFromState(
                control_timestep=self.ctrl_dt, state=self._state(i), target_pos=self.target)
            a = (rpm / self.envs[i].HOVER_RPM - 1.0) / self.rpm_scale
            out.append(a if self.act == 'rpm' else a.mean(keepdims=True))
        return np.clip(np.asarray(out, np.float32), -1, 1)

    def step(self, actions):
        """actions (B, A). Returns obs (B,12), reward (B,), done (B,) and auto-resets."""
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        for i, e in enumerate(self.envs):
            _, r, term, trunc, _ = e.step(actions[i].reshape(1, -1))
            self.t[i] += 1
            rew[i], done[i] = r, term or trunc
        dist = np.linalg.norm(self.observe()[:, :3], axis=1)
        for i in np.where(done)[0]:
            self.finished_lengths.append(int(self.t[i]))
            self._reset_one(i)
        return self.observe(), rew, done, dist

    def pop_finished(self):
        out, self.finished_lengths = self.finished_lengths, []
        return out

    def close(self):
        for e in self.envs:
            e.close()


class FakeDroneBatch:
    """Dependency-free stand-in with the DroneBatch API (1-D altitude dynamics,
    PD teacher). Only for smoke tests of the training loop."""

    def __init__(self, n_envs=2, act='one_d_rpm', ctrl_freq=50, episode_sec=2, seed=0, **_):
        self.n, self.act = n_envs, act
        self.n_actions = 4 if act == 'rpm' else 1
        self.dt = 1.0 / ctrl_freq
        self.max_steps = int(episode_sec * ctrl_freq)
        self.rng = np.random.default_rng(seed)
        self.z = np.zeros(n_envs); self.v = np.zeros(n_envs)
        self.t = np.zeros(n_envs, int); self.finished_lengths = []

    def _reset_one(self, i):
        self.z[i], self.v[i], self.t[i] = self.rng.uniform(-0.3, 0.3), 0.0, 0

    def reset(self):
        for i in range(self.n):
            self._reset_one(i)
        return self.observe()

    def observe(self):
        o = np.zeros((self.n, 12), np.float32)
        o[:, 2], o[:, 8] = self.z, self.v
        return o

    def teacher(self):
        a = np.clip(-2.0 * self.z - 1.0 * self.v, -1, 1)
        return np.repeat(a[:, None], self.n_actions, 1).astype(np.float32)

    def step(self, actions):
        acc = 10.0 * np.asarray(actions).mean(1)
        self.v += acc * self.dt; self.z += self.v * self.dt; self.t += 1
        done = (np.abs(self.z) > 1.5) | (self.t >= self.max_steps)
        dist = np.abs(self.z).copy()
        for i in np.where(done)[0]:
            self.finished_lengths.append(int(self.t[i])); self._reset_one(i)
        return self.observe(), -dist, done, dist

    def pop_finished(self):
        out, self.finished_lengths = self.finished_lengths, []
        return out

    def close(self):
        pass
