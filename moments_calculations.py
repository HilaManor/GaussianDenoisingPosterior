import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, NamedTuple, Union
import time


class Moments(NamedTuple):
    vmu1: np.array
    vmu2: np.array
    vmu3: np.array
    vmu4: np.array
    vmu4_other: np.array


def get_eigvecs(model: torch.nn.Module,
                nim: torch.Tensor,
                mask: torch.Tensor,
                n_ev: int,
                sigma: float,
                device: torch.device,
                c: float = 1e-6,
                iters: int = 5,
                double_precision: bool = False,
                compare_backprop: bool = False) -> Union[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Optional[List]], torch.Tensor,
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Optional[List], torch.Tensor,
                          List[Tuple[float, float]], torch.Tensor]]:
    r"""Calculate the eigvecs of the covariance matrix using power iterations
    $\mu_1(y + \epsilon) - \mu_1(y) ~ \frac{d\mu_1(y)}{dy}\cdot\epsilon$$

    :param torch.nn.Module model: Denoising model
    :param torch.Tensor nim: Noisy image to pass in the model
    :param torch.Tensor mask: Mask of the patch for which to calculate eigenvectors
    :param int n_ev: amount of eigvecs to calculate
    :param float sigma: non-blind sigma of the gaussian noise
    :param torch.device device: cuda:int/cpu
    :param float c: normalizing constant for the 1st derivative linear approximation, defaults to 1e-6
    :param int iters: amount of power iterations to preform, defaults to 5
    :param float skew_c: _description_, defaults to 1e-5
    :param float kurt_c: _description_, defaults to 1e-5
    :param bool double_precision: _description_, defaults to False
    :param bool use_poly: _description_, defaults to False
    :return _type_: _description_
    """

    nim = nim.to(device)
    with torch.no_grad():
        im_mmse = _forward_directional(model, nim.unsqueeze(0), device, 0, 0, double_precision).squeeze(0)

    print(f'MSE (x_hat, y): {torch.nn.functional.mse_loss(im_mmse, nim)}')
    print(f'sqrt (MSE (x_hat, y)): {torch.nn.functional.mse_loss(im_mmse, nim).sqrt()}')
    print(f'assumed noise sigma: {sigma}')
    if nim.min() < 0:  # DDPM_FFHQ
        print('sqrt (MSE (x_hat, y)) - in [0,1]: sqrt (MSE (x_hat, y)):'
              f'{torch.nn.functional.mse_loss(im_mmse, nim).sqrt() / 2 * 255}')
        print(f'assumed noise sigma - in [0,1]: {sigma / 2 * 255}')

    if sigma is None:
        sigma = torch.nn.functional.mse_loss(im_mmse, nim).sqrt().item()
    print(f'Using assumed noise sigma: {sigma}')
    b_masked_mmse = (im_mmse * mask).repeat(n_ev, *[1]*len(nim.shape)).to(device)
    # del im_mmse
    torch.cuda.empty_cache()
    nim = nim.repeat(n_ev, *[1]*len(nim.shape)).to(device)
    eigvecs = torch.randn(*nim.shape, device=device, dtype=nim.dtype) * mask * c
    prev = eigvecs.clone()
    # prevord = eigvecs.clone()
    for i, p in enumerate(prev):
        prev[i] = p / p.norm()
    corr = []
    # corr = None
    # loss = []
    cosine_similarity = []
    timing = []
    # lossord = []
    # for _ in range(iters):
    for i in tqdm(range(iters)):
        if compare_backprop:
            eigvecs, jacprod_norm, eigvecs_backprop, jacobprod_backprop_norm, (apprx_time, bp_time) = _power_iteration(
                model, nim, n_ev, device, b_masked_mmse, eigvecs, double_precision, mask, compare_backprop=True, c=c)
            timing.append((apprx_time, bp_time))
        else:
            eigvecs, jacprod_norm = _power_iteration(model, nim, n_ev, device, b_masked_mmse, eigvecs,
                                                     double_precision, mask)
        _, tmp = (jacprod_norm / c * (sigma**2)).reshape(n_ev, ).sort(descending=True, stable=True)
        eigvecs = eigvecs[tmp, ...]
        # next = eigvecs.clone()
        # next = eigvecs[tmp, ...].clone()
        # loss.append((prev - eigvecs).norm()/ n_ev)
        # lossord.append((prevord - next).norm() / n_ev)
        ev1 = eigvecs.cpu()[0].ravel()
        ev2 = eigvecs.cpu()[1].ravel()
        ev3 = eigvecs.cpu()[2].ravel()
        evs_now = torch.stack([ev1, ev2, ev3], dim=0)
        ev1 = prev.cpu()[0].ravel()
        ev2 = prev.cpu()[1].ravel()
        ev3 = prev.cpu()[2].ravel()
        evs_prev = torch.stack([ev1, ev2, ev3], dim=0)
        corr.append((evs_now @ evs_prev.T).cpu().numpy())
        prev = eigvecs.clone()
        # prevord = next.clone()

        if compare_backprop:
            cosine_similarity.append((eigvecs * eigvecs_backprop).sum().cpu())

        eigvecs *= c  # This makes sure that in the next iteration the approximation will still hold
    eigvals = (jacprod_norm / c * (sigma**2)).reshape(n_ev, )
    eigvecs /= c
    if compare_backprop:
        eigvals = (jacobprod_backprop_norm * (sigma**2)).reshape(n_ev, )
    eigvals, indices = eigvals.sort(descending=True, stable=True)  # TODO ?? swap places??
    eigvecs = eigvecs[indices, ...]  # TODO ?????? swap places??

    if compare_backprop:
        return eigvecs, eigvals, b_masked_mmse[0], sigma, corr, im_mmse, cosine_similarity, timing, eigvecs_backprop

    return eigvecs, eigvals, b_masked_mmse[0], sigma, corr, im_mmse


def calc_moments(model: torch.nn.Module,
                 nim: torch.Tensor,
                 mask: torch.Tensor,
                 sigma: float,
                 device: torch.device,
                 masked_mmse: torch.Tensor,
                 eigvecs: torch.Tensor,
                 eigvals: torch.Tensor,
                 var_c: float = 1e-5, skew_c: float = 1e-5, kurt_c: float = 1e-5,
                 use_poly: bool = False, poly_deg: int = 6, poly_bound: float = 1, poly_pts: int = 30,
                 double_precision: bool = False
                 ) -> Moments:
    print("Computing moments...")
    n_ev = len(eigvecs)
    with torch.no_grad():
        b_masked_mmse = masked_mmse.repeat(n_ev, *[1]*len(nim.shape)).to(device)
        nim = nim.repeat(n_ev, *[1]*len(nim.shape)).to(device)

        first_moments = (eigvecs.reshape(n_ev, -1) * b_masked_mmse.reshape(n_ev, -1)).sum(axis=1)
        print(f'first_moments: {first_moments}')

        second_moments, third_moments, fourth_moments = _calc_second_third_and_fourth_moments(
            model, nim, n_ev, device, var_c, skew_c, kurt_c, b_masked_mmse, eigvecs, sigma,
            mask, double_precision, use_poly, poly_deg, poly_bound, poly_pts)
        
        # third_moments, fourth_moments = _calc_third_and_fourth_moments(model, nim, n_ev, device,
        #                                                                skew_c, kurt_c, b_masked_mmse,
        #                                                                second_moments,
        #                                                                eigvecs, sigma, mask, double_precision,
        #                                                                use_poly, poly_deg, poly_bound, poly_pts)
        more_fourth_moments = _calc_fourth_moments(model, nim, n_ev, device, 10*kurt_c, b_masked_mmse, eigvecs,
                                                   second_moments, sigma, mask, double_precision)

        print(f'third_moments: {third_moments}')
        print(f'fourth_moments: {fourth_moments} \t or {more_fourth_moments}')

    return Moments(vmu1=first_moments.cpu().numpy().astype(np.float64),
                   vmu2=second_moments.cpu().numpy().astype(np.float64),
                   vmu3=third_moments.cpu().numpy().astype(np.float64),
                   vmu4=fourth_moments.cpu().numpy().astype(np.float64),
                   vmu4_other=more_fourth_moments.cpu().numpy().astype(np.float64),
                   )


def _forward_directional(model: torch.nn.Module,
                         patch: torch.Tensor, device: torch.device,
                         eigvecs: torch.Tensor, amount: float, double_precision: bool = False):
    with torch.no_grad():
        input = patch + amount * eigvecs
        if double_precision:
            input = input.to(torch.double).to(device)
        output = model(input)
    return output


def _power_iteration(model: torch.nn.Module, nim: torch.Tensor,
                     n_ev: int, device: torch.device,
                     b_masked_mmse: torch.Tensor, eigvecs: torch.Tensor, double_precision: bool,
                     mask: torch.Tensor,
                     compare_backprop: bool = False,
                     c: Optional[float] = None) -> Union[
                         Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[float, float]],
                         Tuple[torch.Tensor, torch.Tensor]]:

    torch.cuda.synchronize()
    start_backprop = time.time()
    if compare_backprop:

        nimf = nim[0].clone().unsqueeze(0)  # .to(torch.float32)
        nimf.requires_grad_(True)
        x = nimf.to(torch.double).to(device)
        y = model(x)
        yy = y * mask

        if n_ev > 1:
            raise NotImplementedError("TODO")

            grads = []
            for i in range(n_ev - 1):
                z = (yy * (eigvecs[i].unsqueeze(0)/c)).sum(dim=(1, 2, 3))
                z.backward(retain_graph=True)
                grads.append(nimf.grad[0].clone())
            z = (yy * (eigvecs[n_ev-1].unsqueeze(0)/c)).sum(dim=(1, 2, 3))
            z.backward()
            grads.append(nimf.grad[0].clone())
            grads = torch.stack(grads)

            if len(nim.shape) == 4:
                permute_arg = (1, 2, 3, 0)
            elif len(nim.shape) == 3:
                permute_arg = (1, 2, 0)
            elif len(nim.shape) == 2:
                permute_arg = (1, 0)
            norm_of_grads = grads[:, mask.to(torch.bool)].norm(dim=1)
            # This completes the power iteration
            eigvecs_backprop = (grads / norm_of_grads.reshape(n_ev, *[1]*(len(nim.shape) - 1))) * mask

            # Now just make sure the evs are orthonormal
            Q, _ = torch.linalg.qr(eigvecs_backprop.permute(*permute_arg).reshape(-1, n_ev), mode='reduced')
            eigvecs_backprop = Q / Q.norm(dim=0)
            eigvecs_backprop = eigvecs_backprop.T.reshape(grads.shape)
        else:
            z = (yy * (eigvecs/c)).sum(dim=(1, 2, 3))
            z.backward()
            grads = nimf.grad.clone()

            norm_of_grads = grads[mask.unsqueeze(0).to(torch.bool)].norm()
            eigvecs_backprop = (grads / norm_of_grads) * mask  # This completes the power iteration
    torch.cuda.synchronize()
    bp_time = time.time() - start_backprop

    torch.cuda.synchronize()
    start_approx = time.time()

    with torch.no_grad():
        unmaksed_out = _forward_directional(model, nim, device, eigvecs, 1, double_precision)
        out = unmaksed_out * mask
        Ab = out - b_masked_mmse
        if n_ev > 1:
            if len(nim.shape) == 4:
                permute_arg = (1, 2, 3, 0)
            elif len(nim.shape) == 3:
                permute_arg = (1, 2, 0)
            elif len(nim.shape) == 2:
                permute_arg = (1, 0)
            norm_of_Ab = Ab[:, mask.to(torch.bool)].norm(dim=1)
            # Now complete the power iteration:
            eigvecs = (Ab / norm_of_Ab.reshape(n_ev, *[1]*(len(nim.shape) - 1))) * mask

            # Now just make sure the evs are orthonormal
            Q, R = torch.linalg.qr(eigvecs.permute(*permute_arg).reshape(-1, n_ev), mode='reduced')
            # TODO - add determinant check
            swap = torch.prod(torch.linalg.diagonal(R))
            if swap < 0:  # TODO VERIFY
                Q *= -1

            eigvecs = Q / Q.norm(dim=0)
            eigvecs = eigvecs.T.reshape(Ab.shape)
        else:
            norm_of_Ab = Ab[mask.unsqueeze(0).to(torch.bool)].norm()
            eigvecs = (Ab / norm_of_Ab) * mask  # This completes the power iteration
    torch.cuda.synchronize()
    apprx_time = time.time() - start_approx

    if compare_backprop:
        return eigvecs, norm_of_Ab, eigvecs_backprop, norm_of_grads, (apprx_time, bp_time)

    return eigvecs, norm_of_Ab


def _calc_fourth_moments(model: torch.nn.Module,
                         nim: torch.Tensor, n_ev: int, device: torch.device,
                         c: float, b_masked_mmse: torch.Tensor, eigvecs: torch.Tensor,
                         second_moments: torch.Tensor, sigma: float, mask: torch.Tensor, double_precision: bool):
    first_term = _forward_directional(model, nim, device, eigvecs, 1.5 * c, double_precision) * mask
    second_term = _forward_directional(model, nim, device, eigvecs, 0.5 * c, double_precision) * mask
    # third_term = b_masked_mmse
    third_term = _forward_directional(model, nim, device, eigvecs, -0.5 * c, double_precision) * mask
    fourth_term = _forward_directional(model, nim, device, eigvecs, -1.5 * c, double_precision) * mask

    deriv_approx = (1/(c**3)) * (first_term - 3 * second_term + 3 * third_term - fourth_term)
    # deriv_approx = (1/(c**3)) * (first_term - 3 * second_term + 3 * third_term - fourth_term)
    fourth_moments = (sigma ** 6) * (eigvecs.reshape(n_ev, -1) * deriv_approx.reshape(n_ev, -1)).sum(axis=1) + \
        3 * (second_moments ** 2)
    return fourth_moments


def _calc_third_moments(model: torch.nn.Module, nim: torch.Tensor, n_ev: int, device: torch.device,
                        c: float, bmmse: torch.Tensor, eigvecs: torch.Tensor, sigma: float,
                        mask: torch.Tensor, double_precision: bool):
    mid_term = bmmse
    left_term = _forward_directional(model, nim, device, eigvecs, -c, double_precision) * mask
    right_term = _forward_directional(model, nim, device, eigvecs, c, double_precision) * mask

    third_moments = (1 / (c**2)) * (sigma ** 4) * (
        eigvecs.reshape(n_ev, -1) * (left_term - 2 * mid_term + right_term).reshape(n_ev, -1)).sum(axis=1)
    return third_moments


def _calc_third_and_fourth_moments(model: torch.nn.Module, nim: torch.Tensor,
                                   n_ev: int, device: torch.device,
                                   skew_c: float, kurt_c: float,
                                   b_masked_mmse: torch.Tensor, second_moments: torch.Tensor,
                                   eigvecs: torch.Tensor, sigma: float,
                                   mask: torch.Tensor, double_precision: bool,
                                   use_poly: bool, poly_deg: int,
                                   poly_bound: float, poly_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_poly:
        third_moments, fourth_moments = _calc_third_and_fourth_moments_by_polyfit(model, nim, n_ev, device,
                                                                                  second_moments, eigvecs,
                                                                                  sigma, mask, double_precision,
                                                                                  poly_deg, poly_bound, poly_pts)
    elif kurt_c == skew_c:
        c = kurt_c
        first_term = _forward_directional(model, nim, device, eigvecs, 2 * c, double_precision) * mask
        second_term = _forward_directional(model, nim, device, eigvecs, c, double_precision) * mask
        third_term = b_masked_mmse
        fourth_term = _forward_directional(model, nim, device, eigvecs, -c, double_precision) * mask

        deriv_approx = (1/(c**3)) * (first_term - 3 * second_term + 3 * third_term - fourth_term)
        fourth_moments = (sigma ** 6) * (eigvecs.reshape(n_ev, -1) * deriv_approx.reshape(n_ev, -1)).sum(axis=1) + \
            3 * (second_moments ** 2)

        third_moments = (1 / (c**2)) * (sigma ** 4) * (eigvecs.reshape(n_ev, -1) *
                                                       (second_term - 2 * third_term + fourth_term).reshape(n_ev, -1)
                                                       ).sum(axis=1)
    else:
        third_moments = _calc_third_moments(model, nim, n_ev, device, skew_c, b_masked_mmse, eigvecs, sigma, mask,
                                            double_precision)
        fourth_moments = _calc_fourth_moments(model, nim, n_ev, device, kurt_c, b_masked_mmse, eigvecs, second_moments,
                                              sigma, mask, double_precision)
    return third_moments, fourth_moments


def _calc_second_third_and_fourth_moments(model: torch.nn.Module, nim: torch.Tensor,
                                          n_ev: int, device: torch.device,
                                          var_c: float, skew_c: float, kurt_c: float,
                                          b_masked_mmse: torch.Tensor,
                                          eigvecs: torch.Tensor, sigma: float,
                                          mask: torch.Tensor, double_precision: bool,
                                          use_poly: bool, poly_deg: int,
                                          poly_bound: float, poly_pts: int) -> Tuple[torch.Tensor, torch.Tensor]:

    # Second moment must be calculated via linear approx.
    second_term = _forward_directional(model, nim, device, eigvecs, 0.5 * var_c, double_precision) * mask
    third_term = _forward_directional(model, nim, device, eigvecs, -0.5 * var_c, double_precision) * mask
    second_moments = (1 / var_c) * (sigma ** 2) * (eigvecs.reshape(n_ev, -1) *
                                                   (second_term - third_term).reshape(n_ev, -1)
                                                   ).sum(axis=1)
    # second_moments = eigvals
    print(f'second_moments: {second_moments}')

    if use_poly:
        third_moments, fourth_moments = _calc_third_and_fourth_moments_by_polyfit(model, nim, n_ev, device,
                                                                                  second_moments, eigvecs,
                                                                                  sigma, mask, double_precision,
                                                                                  poly_deg, poly_bound, poly_pts)
    else:
        third_moments = _calc_third_moments(model, nim, n_ev, device, skew_c, b_masked_mmse, eigvecs, sigma, mask,
                                            double_precision)
        if kurt_c == var_c:
            first_term = _forward_directional(model, nim, device, eigvecs, 1.5 * kurt_c, double_precision) * mask
            fourth_term = _forward_directional(model, nim, device, eigvecs, -1.5 * kurt_c, double_precision) * mask

            deriv_approx = (1/(kurt_c**3)) * (first_term - 3 * second_term + 3 * third_term - fourth_term)
            fourth_moments = (sigma ** 6) * (eigvecs.reshape(n_ev, -1) * deriv_approx.reshape(n_ev, -1)).sum(axis=1) + \
                3 * (second_moments ** 2)
        else:
            fourth_moments = _calc_fourth_moments(model, nim, n_ev, device, kurt_c, b_masked_mmse, eigvecs,
                                                  second_moments, sigma, mask, double_precision)

    return second_moments, third_moments, fourth_moments


def _calc_third_and_fourth_moments_by_polyfit(model: torch.nn.Module,
                                              nim: torch.Tensor, n_ev: int, device: torch.device,
                                              second_moments: torch.Tensor,
                                              eigvecs: torch.Tensor,
                                              sigma: float, mask: torch.Tensor,
                                              double_precision: bool,
                                              poly_deg: int = 6, poly_bound: float = 1, poly_pts: int = 30):
    if poly_deg < 4:
        raise ValueError("We want to find the third directional derivative")

    def d2_dx_poly(xs, coeffs):
        deg = len(coeffs) - 1

        value = np.zeros_like(xs, dtype=xs.dtype)
        for i in range(deg - 1):
            value += (deg - i) * (deg - 1 - i) * coeffs[i] * (xs ** (deg - 2 - i))
        return value

    def d3_dx_poly(xs, coeffs):
        deg = len(coeffs) - 1

        value = np.zeros_like(xs, dtype=xs.dtype)
        for i in range(deg - 2):
            value += (deg - i) * (deg - 1 - i) * (deg - 2 - i) * coeffs[i] * (xs ** (deg - 3 - i))
        return value

    third_moments = torch.zeros(n_ev, device=device)
    fourth_moments = torch.zeros(n_ev, device=device)
    poly_xs = torch.linspace(-poly_bound, poly_bound, poly_pts, device=device) * second_moments.sqrt().reshape(-1, 1)
    poly_xs = (torch.logspace(0, np.log10(poly_bound + 1), poly_pts//2, device=device) - 1) * second_moments.sqrt().reshape(-1, 1)
    poly_xs = torch.concatenate([-1 * poly_xs.flip(dims=(-1,))[:, :-1], poly_xs], dim=-1)
    poly_pts = poly_xs.shape[-1]
    for i in tqdm(range(n_ev)):
        batched_eigvs = (eigvecs[i].ravel() * poly_xs[i].reshape(-1, 1)).reshape(poly_pts, *eigvecs[i].shape)
        batched_inps = nim[0].repeat(poly_pts, *[1]*(len(nim.shape) - 1)).to(device) + batched_eigvs
        batched_outs = []
        for b in range(batched_inps.shape[0]):
            batched_outs.append(_forward_directional(model, batched_inps[b].unsqueeze(0), device, 0, 0, double_precision
                                                     ).detach())
        batched_outs = torch.stack(batched_outs, dim=0)
        poly_ys = (eigvecs[i].reshape(1, -1) @ (batched_outs * mask).reshape(poly_pts, -1).T).T
        coeffs = np.polyfit(poly_xs[i].cpu().numpy(), poly_ys.cpu().numpy(), deg=poly_deg)

        d2_dx = d2_dx_poly(np.array([0], dtype=poly_ys.cpu().numpy().dtype), coeffs)
        d3_dx = d3_dx_poly(np.array([0], dtype=poly_ys.cpu().numpy().dtype), coeffs)
        third_moments[i] = (sigma ** 4) * d2_dx[0]
        fourth_moments[i] = (sigma ** 6) * d3_dx[0] + 3 * (second_moments[i] ** 2)
    return third_moments, fourth_moments
