"""Behavioral checks: python -m unittest test_shared_lstm -v."""

import unittest

import torch

from shared_lstm import SharedLSTMEncoder, detach_state


class SharedLSTMTests(unittest.TestCase):
    def test_chunks_match_full_sequence(self):
        for inputs, hidden in ((4, 4), (3, 7), (6, 2)):
            model = SharedLSTMEncoder(inputs, hidden).double().eval()
            x = torch.randn(2, 5, inputs, dtype=torch.float64)
            full, final = model(x)
            first, state = model(x[:, :1])
            rest, continued = model(x[:, 1:], state)
            self.assertEqual(tuple(first.shape), (2, 1, hidden))
            self.assertEqual(tuple(state[0].shape), (1, 2, hidden))
            self.assertEqual(tuple(state[1].shape), (1, 2, hidden))
            torch.testing.assert_close(torch.cat((first, rest), dim=1), full)
            for actual, expected in zip(continued, final):
                torch.testing.assert_close(actual, expected)
            # A fresh call must not inherit the prior call's state.
            torch.testing.assert_close(model(x)[0], full)

    def test_gradient_continuity_and_explicit_detach(self):
        model = SharedLSTMEncoder().double()
        x = torch.randn(2, 1, 4, dtype=torch.float64, requires_grad=True)
        h = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
        c = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
        _, state = model(x, (h, c))
        next_x = torch.randn(2, 2, 4, dtype=torch.float64)
        model(next_x, state)[0].sum().backward()
        for value in (x, h, c):
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0)
        detached = detach_state(state)
        for old, new in zip(state, detached):
            torch.testing.assert_close(old, new)
            self.assertFalse(new.requires_grad)
            self.assertIsNone(new.grad_fn)
        x.grad = h.grad = c.grad = None
        model(next_x, detached)[0].sum().backward()
        for value in (x, h, c):
            self.assertIsNone(value.grad)

    def test_configuration_validation(self):
        for value in (0, -1, True, 1.5, "4"):
            for argument in ("input_size", "hidden_size"):
                with self.subTest(value=value, argument=argument):
                    with self.assertRaises((TypeError, ValueError)):
                        SharedLSTMEncoder(**{argument: value})

    def test_input_and_state_validation(self):
        model = SharedLSTMEncoder()
        x = torch.zeros(2, 1, 4)
        good = torch.zeros(1, 2, 4)
        bad_inputs = (None, torch.zeros(2, 4), torch.zeros(2, 1, 3),
                      torch.zeros(2, 0, 4), torch.zeros(0, 1, 4),
                      x.long(), x.double(), torch.empty(2, 1, 4, device="meta"))
        for bad in bad_inputs:
            with self.assertRaises((TypeError, ValueError)):
                model(bad)
        bad_states = ((good,), [good, good], (None, good),
                      (good, torch.zeros(2, 2, 4)),
                      (torch.zeros(1, 3, 4), good), (good.double(), good),
                      (good, torch.empty(1, 2, 4, device="meta")))
        for bad in bad_states:
            with self.assertRaises((TypeError, ValueError)):
                model(x, bad)

    def test_standard_module_behavior(self):
        model = SharedLSTMEncoder(3, 5).to(dtype=torch.float64)
        x = torch.randn(2, 3, 3, dtype=torch.float64)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        model.train()
        expected = model(x)[0]
        model.eval()
        torch.testing.assert_close(model(x)[0], expected)
        clone = SharedLSTMEncoder(3, 5).double()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(clone(x)[0], expected)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        with torch.no_grad():
            self.assertFalse(model(x)[0].requires_grad)
        self.assertTrue(expected.requires_grad)
        self.assertEqual(sum(isinstance(m, torch.nn.LSTM) for m in model.modules()), 1)
        self.assertEqual(model.lstm.num_layers, 1)
        self.assertTrue(model.lstm.batch_first)
        self.assertFalse(model.lstm.bidirectional)
        self.assertEqual(model.lstm.proj_size, 0)
        self.assertEqual(model.lstm.dropout, 0)


if __name__ == "__main__":
    unittest.main()
