"""The linear-chain CRF used by the optional BiLSTM-CRF run."""

import torch
from torch import Tensor, nn


class LinearChainCRF(nn.Module):
    """Score tag sequences while ignoring the padding tag."""

    def __init__(self, tag_count: int, padding_tag_id: int):
        super().__init__()
        self.tag_count = tag_count
        self.padding_tag_id = padding_tag_id
        self.transitions = nn.Parameter(torch.zeros(tag_count, tag_count))
        self.start_transitions = nn.Parameter(torch.zeros(tag_count))
        self.end_transitions = nn.Parameter(torch.zeros(tag_count))

    def _valid_parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        """Mask padding from transitions and sentence boundaries."""
        valid = torch.ones(
            self.tag_count,
            self.tag_count,
            dtype=torch.bool,
            device=self.transitions.device,
        )
        valid[self.padding_tag_id, :] = False
        valid[:, self.padding_tag_id] = False
        transitions = self.transitions.masked_fill(~valid, -1e4)
        start = self.start_transitions.clone()
        end = self.end_transitions.clone()
        start[self.padding_tag_id] = -1e4
        end[self.padding_tag_id] = -1e4
        return transitions, start, end

    def _sequence_score(
        self,
        emissions: Tensor,
        tags: Tensor,
        length: int,
    ) -> Tensor:
        transitions, start, end = self._valid_parameters()
        score = start[tags[0]] + emissions[0, tags[0]]
        for position in range(1, length):
            score = (
                score
                + transitions[tags[position - 1], tags[position]]
                + emissions[position, tags[position]]
            )
        return score + end[tags[length - 1]]

    def _log_partition(self, emissions: Tensor, length: int) -> Tensor:
        transitions, start, end = self._valid_parameters()
        scores = start + emissions[0]
        for position in range(1, length):
            scores = torch.logsumexp(
                scores.unsqueeze(1)
                + transitions
                + emissions[position].unsqueeze(0),
                dim=0,
            )
        return torch.logsumexp(scores + end, dim=0)

    def neg_log_likelihood(
        self,
        emissions: Tensor,
        tags: Tensor,
        sentence_lengths: Tensor,
    ) -> Tensor:
        """Return the mean negative log-likelihood for the batch."""
        losses = []
        for sentence_index, length_tensor in enumerate(sentence_lengths):
            length = int(length_tensor.item())
            if length < 1:
                continue
            losses.append(
                self._log_partition(emissions[sentence_index], length)
                - self._sequence_score(
                    emissions[sentence_index],
                    tags[sentence_index],
                    length,
                )
            )
        if not losses:
            raise ValueError("sentence_lengths must contain a real token")
        return torch.stack(losses).mean()

    def decode(self, emissions: Tensor, sentence_lengths: Tensor) -> Tensor:
        """Find the best tag path for each sentence."""
        transitions, start, end = self._valid_parameters()
        decoded = torch.full(
            emissions.shape[:2],
            self.padding_tag_id,
            dtype=torch.long,
            device=emissions.device,
        )
        for sentence_index, length_tensor in enumerate(sentence_lengths):
            length = int(length_tensor.item())
            if length < 1:
                continue
            scores = start + emissions[sentence_index, 0]
            backpointers = []
            for position in range(1, length):
                next_scores = scores.unsqueeze(1) + transitions
                best_scores, best_tags = next_scores.max(dim=0)
                scores = best_scores + emissions[sentence_index, position]
                backpointers.append(best_tags)
            last_tag = (scores + end).argmax()
            path = [last_tag]
            for best_tags in reversed(backpointers):
                last_tag = best_tags[last_tag]
                path.append(last_tag)
            decoded[sentence_index, :length] = torch.stack(path[::-1])
        return decoded
