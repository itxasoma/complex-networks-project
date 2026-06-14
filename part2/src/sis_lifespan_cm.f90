!=============================================================
! sis_lifespan_cm.f90
!
! Part 2 code:
! - power-law degree sequence with kmin=4
! - simple configuration model
! - CSR adjacency + reverse-stub map
! - SIS Gillespie
! - lifespan method: P_end and tau for finite runs
!
! Compile:
!   gfortran -O3 -std=f2008 -Wall -Wextra -o sis_lifespan_cm sis_lifespan_cm.f90
!
! Run:
!   ./sis_lifespan_cm 100000 3.5 2000 0.02 0.10 25 results_g35_N1e5.dat
!
! Args:
!   1: N
!   2: gamma
!   3: nruns
!   4: lambda_min
!   5: lambda_max
!   6: nlambda
!   7: output_file
!
! Output columns:
!   lambda   tau_non_endemic   P_end   N   gamma   nruns   M
!=============================================================

module rng_mod
  use iso_fortran_env, only: real64
  implicit none
contains
  subroutine seed_rng(offset)
    integer, intent(in), optional :: offset
    integer :: n, i, clk, off
    integer, allocatable :: seed(:)

    off = 0
    if (present(offset)) off = offset

    call random_seed(size=n)
    allocate(seed(n))
    call system_clock(count=clk)

    do i = 1, n
      seed(i) = mod(clk + 104729*i + 8191*off, 2147483646)
      if (seed(i) <= 0) seed(i) = i + 37
    enddo

    call random_seed(put=seed)
    deallocate(seed)
  end subroutine seed_rng

  real(real64) function urand()
    call random_number(urand)
    if (urand <= 0.0_real64) urand = 1.0e-12_real64
  end function urand
end module rng_mod

module sort_mod
  use iso_fortran_env, only: int64
  implicit none
contains
  recursive subroutine quicksort_i64(a, left, right)
    integer(int64), intent(inout) :: a(:)
    integer(int64), intent(in)    :: left, right
    integer(int64) :: i, j, pivot, tmp

    if (left >= right) return

    i = left
    j = right
    pivot = a((left + right)/2)

    do
      do while (a(i) < pivot)
        i = i + 1
      enddo
      do while (a(j) > pivot)
        j = j - 1
      enddo
      if (i <= j) then
        tmp = a(i)
        a(i) = a(j)
        a(j) = tmp
        i = i + 1
        j = j - 1
      endif
      if (i > j) exit
    enddo

    if (left < j) call quicksort_i64(a, left, j)
    if (i < right) call quicksort_i64(a, i, right)
  end subroutine quicksort_i64
end module sort_mod

module cm_mod
  use iso_fortran_env, only: int32, int64, real64
  use rng_mod
  use sort_mod
  implicit none
contains
  subroutine generate_powerlaw_degrees(n, gamma, kmin, deg)
    integer(int64), intent(in)  :: n
    real(real64), intent(in)    :: gamma
    integer, intent(in)         :: kmin
    integer(int32), intent(out) :: deg(n)

    integer(int64) :: i
    real(real64)   :: u, expo, x

    expo = 1.0_real64 / (gamma - 1.0_real64)

    do i = 1, n
      u = urand()
      x = real(kmin, real64) * (1.0_real64 - u)**(-expo)
      deg(i) = max(kmin, int(x, int32))
    enddo

    if (mod(sum(int(deg, int64)), 2_int64) /= 0_int64) deg(1) = deg(1) + 1
  end subroutine generate_powerlaw_degrees

  subroutine shuffle_i32(a)
    integer(int32), intent(inout) :: a(:)
    integer(int64) :: i, j
    integer(int32) :: tmp

    do i = size(a, kind=int64), 2_int64, -1_int64
      j = 1_int64 + int(urand() * real(i, real64), int64)
      tmp = a(i)
      a(i) = a(j)
      a(j) = tmp
    enddo
  end subroutine shuffle_i32

  pure integer(int64) function edge_key(u, v, n)
    integer(int32), intent(in) :: u, v
    integer(int64), intent(in) :: n
    edge_key = int(u - 1, int64) * n + int(v, int64)
  end function edge_key

  subroutine decode_key(key, n, u, v)
    integer(int64), intent(in)  :: key, n
    integer(int32), intent(out) :: u, v
    integer(int64) :: uu, vv
    uu = (key - 1_int64) / n + 1_int64
    vv = key - (uu - 1_int64) * n
    u = int(uu, int32)
    v = int(vv, int32)
  end subroutine decode_key

  subroutine build_erased_cm_csr(n, deg_target, deg, rowptr, colind, rev, src, m)
    integer(int64), intent(in)              :: n
    integer(int32), intent(in)              :: deg_target(n)
    integer(int32), allocatable, intent(out):: deg(:), colind(:), src(:)
    integer(int64), allocatable, intent(out):: rowptr(:), rev(:)
    integer(int64), intent(out)             :: m

    integer(int64) :: stubN, i, p, e0, euniq
    integer(int32), allocatable :: stubs(:)
    integer(int64), allocatable :: keys(:), uniq(:), pos(:)
    integer(int32) :: u, v

    stubN = sum(int(deg_target, int64))
    allocate(stubs(stubN))

    p = 0_int64
    do i = 1, n
      do v = 1, deg_target(i)
        p = p + 1
        stubs(p) = int(i, int32)
      enddo
    enddo

    call shuffle_i32(stubs)

    allocate(keys(stubN/2))
    e0 = 0_int64
    do i = 1, stubN-1, 2
      u = stubs(i)
      v = stubs(i+1)
      if (u /= v) then
        if (u < v) then
          e0 = e0 + 1
          keys(e0) = edge_key(u, v, n)
        else
          e0 = e0 + 1
          keys(e0) = edge_key(v, u, n)
        endif
      endif
    enddo
    deallocate(stubs)

    if (e0 == 0) stop "No edges after pairing."

    call quicksort_i64(keys, 1_int64, e0)

    allocate(uniq(e0))
    euniq = 1_int64
    uniq(1) = keys(1)
    do i = 2, e0
      if (keys(i) /= keys(i-1)) then
        euniq = euniq + 1
        uniq(euniq) = keys(i)
      endif
    enddo
    deallocate(keys)

    m = euniq
    allocate(deg(n))
    deg = 0

    do i = 1, m
      call decode_key(uniq(i), n, u, v)
      deg(u) = deg(u) + 1
      deg(v) = deg(v) + 1
    enddo

    allocate(rowptr(n+1), pos(n))
    rowptr(1) = 1_int64
    do i = 1, n
      rowptr(i+1) = rowptr(i) + int(deg(i), int64)
    enddo

    allocate(colind(2*m), src(2*m), rev(2*m))
    pos = rowptr(1:n)

    do i = 1, m
      call decode_key(uniq(i), n, u, v)

      p = pos(u)
      colind(p) = v
      src(p) = u

      colind(pos(v)) = u
      src(pos(v)) = v

      rev(p) = pos(v)
      rev(pos(v)) = p

      pos(u) = pos(u) + 1_int64
      pos(v) = pos(v) + 1_int64
    enddo

    deallocate(pos, uniq)
  end subroutine build_erased_cm_csr
end module cm_mod

module sis_mod
  use iso_fortran_env, only: int32, int64, real64
  use rng_mod
  implicit none
contains
  subroutine add_infected(node, infected, inf_nodes, where_inf, ni)
    integer(int32), intent(in)    :: node
    integer(int32), intent(inout) :: infected(:), inf_nodes(:), where_inf(:)
    integer(int64), intent(inout) :: ni

    if (infected(node) == 1) return
    ni = ni + 1_int64
    inf_nodes(ni) = node
    where_inf(node) = int(ni, int32)
    infected(node) = 1
  end subroutine add_infected

  subroutine remove_infected(node, infected, inf_nodes, where_inf, ni)
    integer(int32), intent(in)    :: node
    integer(int32), intent(inout) :: infected(:), inf_nodes(:), where_inf(:)
    integer(int64), intent(inout) :: ni

    integer(int64) :: idx
    integer(int32) :: lastnode

    if (infected(node) == 0) return

    idx = int(where_inf(node), int64)
    lastnode = inf_nodes(ni)

    inf_nodes(idx) = lastnode
    where_inf(lastnode) = int(idx, int32)

    where_inf(node) = 0
    infected(node) = 0
    ni = ni - 1_int64
  end subroutine remove_infected

  subroutine add_active_stub(stub, act_stub, act_pos, eact)
    integer(int64), intent(in)    :: stub
    integer(int64), intent(inout) :: act_stub(:), act_pos(:)
    integer(int64), intent(inout) :: eact

    if (act_pos(stub) /= 0_int64) return
    eact = eact + 1_int64
    act_stub(eact) = stub
    act_pos(stub) = eact
  end subroutine add_active_stub

  subroutine remove_active_stub(stub, act_stub, act_pos, eact)
    integer(int64), intent(in)    :: stub
    integer(int64), intent(inout) :: act_stub(:), act_pos(:)
    integer(int64), intent(inout) :: eact

    integer(int64) :: idx, laststub

    idx = act_pos(stub)
    if (idx == 0_int64) return

    laststub = act_stub(eact)
    act_stub(idx) = laststub
    act_pos(laststub) = idx
    act_pos(stub) = 0_int64
    eact = eact - 1_int64
  end subroutine remove_active_stub

  subroutine infect_event(stub, rowptr, colind, rev, src, infected, touched, ntouched, &
                          inf_nodes, where_inf, ni, act_stub, act_pos, eact)
    integer(int64), intent(in)    :: stub
    integer(int64), intent(in)    :: rowptr(:), rev(:)
    integer(int32), intent(in)    :: colind(:), src(:)
    integer(int32), intent(inout) :: infected(:), touched(:), inf_nodes(:), where_inf(:)
    integer(int64), intent(inout) :: ntouched, ni, act_stub(:), act_pos(:), eact

    integer(int32) :: u, v, w
    integer(int64) :: q

    u = src(stub)
    v = colind(stub)

    call remove_active_stub(stub, act_stub, act_pos, eact)
    if (infected(v) == 1) return

    call add_infected(v, infected, inf_nodes, where_inf, ni)

    if (touched(v) == 0) then
      touched(v) = 1
      ntouched = ntouched + 1_int64
    endif

    do q = rowptr(v), rowptr(v+1)-1
      w = colind(q)
      if (infected(w) == 1) then
        call remove_active_stub(rev(q), act_stub, act_pos, eact)
      else
        call add_active_stub(q, act_stub, act_pos, eact)
      endif
    enddo
  end subroutine infect_event

  subroutine recover_event(v, rowptr, colind, rev, infected, inf_nodes, where_inf, ni, &
                           act_stub, act_pos, eact)
    integer(int32), intent(in)    :: v
    integer(int64), intent(in)    :: rowptr(:), rev(:)
    integer(int32), intent(in)    :: colind(:)
    integer(int32), intent(inout) :: infected(:), inf_nodes(:), where_inf(:)
    integer(int64), intent(inout) :: ni, act_stub(:), act_pos(:), eact

    integer(int32) :: w
    integer(int64) :: q

    call remove_infected(v, infected, inf_nodes, where_inf, ni)

    do q = rowptr(v), rowptr(v+1)-1
      w = colind(q)
      call remove_active_stub(q, act_stub, act_pos, eact)
      if (infected(w) == 1) then
        call add_active_stub(rev(q), act_stub, act_pos, eact)
      endif
    enddo
  end subroutine recover_event

  subroutine run_lifespan(lambda, cov_thr_nodes, seed_node, rowptr, colind, rev, src, tau, pend)
    real(real64), intent(in)      :: lambda
    integer(int64), intent(in)    :: cov_thr_nodes
    integer(int32), intent(in)    :: seed_node
    integer(int64), intent(in)    :: rowptr(:), rev(:)
    integer(int32), intent(in)    :: colind(:), src(:)
    real(real64), intent(out)     :: tau
    integer, intent(out)          :: pend

    integer(int64) :: n, ni, eact, ntouched, stub
    integer(int32), allocatable :: infected(:), touched(:), inf_nodes(:), where_inf(:)
    integer(int64), allocatable :: act_stub(:), act_pos(:)
    real(real64) :: t, rate, p_inf
    integer(int32) :: v
    !-----------------------------------------------------------------
    ! FIX #1: use a dedicated index variable (idx) for sampling into
    ! act_stub, so the array is dereferenced only once, not twice.
    ! FIX #2: use a dedicated index variable (ridx) for inf_nodes, with
    ! a clamp, preventing the rare out-of-bounds when urand()==1.
    !-----------------------------------------------------------------
    integer(int64) :: idx, ridx

    n = size(rowptr, kind=int64) - 1_int64

    allocate(infected(n), touched(n), inf_nodes(n), where_inf(n))
    allocate(act_stub(size(colind, kind=int64)), act_pos(size(colind, kind=int64)))

    infected = 0
    touched  = 0
    inf_nodes = 0
    where_inf = 0
    act_stub = 0_int64
    act_pos  = 0_int64

    ni = 0_int64
    eact = 0_int64
    ntouched = 0_int64
    t = 0.0_real64
    pend = 0

    call add_infected(seed_node, infected, inf_nodes, where_inf, ni)
    touched(seed_node) = 1
    ntouched = 1_int64

    do stub = rowptr(seed_node), rowptr(seed_node+1)-1
      call add_active_stub(stub, act_stub, act_pos, eact)
    enddo

    do while (ni > 0_int64)
      if (ntouched >= cov_thr_nodes) then
        pend = 1
        exit
      endif

      rate = real(ni, real64) + lambda * real(eact, real64)
      if (rate <= 0.0_real64) exit

      t = t - log(urand()) / rate
      p_inf = lambda * real(eact, real64) / rate

      if (urand() < p_inf) then
        !-----------------------------------------------------------------
        ! FIX #1: compute the index first, clamp it, then read act_stub once.
        ! The old code did: stub = act_stub(...)  /  stub = act_stub(stub)
        ! which dereferenced act_stub twice (double indirection).
        !-----------------------------------------------------------------
        idx = 1_int64 + int(urand() * real(eact, real64), int64)
        if (idx < 1_int64) idx = 1_int64
        if (idx > eact)    idx = eact
        stub = act_stub(idx)
        call infect_event(stub, rowptr, colind, rev, src, infected, touched, ntouched, &
                          inf_nodes, where_inf, ni, act_stub, act_pos, eact)
      else
        !-----------------------------------------------------------------
        ! FIX #2: clamp the recovery node index before accessing inf_nodes.
        !-----------------------------------------------------------------
        ridx = 1_int64 + int(urand() * real(ni, real64), int64)
        if (ridx < 1_int64) ridx = 1_int64
        if (ridx > ni)      ridx = ni
        v = inf_nodes(ridx)
        call recover_event(v, rowptr, colind, rev, infected, inf_nodes, where_inf, ni, &
                           act_stub, act_pos, eact)
      endif
    enddo

    tau = t
    if (pend == 1) tau = 0.0_real64

    deallocate(infected, touched, inf_nodes, where_inf, act_stub, act_pos)
  end subroutine run_lifespan
end module sis_mod


! PROGRAM ==============================================================
program sis_lifespan_cm
  use iso_fortran_env, only: int32, int64, real64
  use rng_mod
  use cm_mod
  use sis_mod
  implicit none

  integer(int64) :: N, M, nruns, i, r, nk4, cov_thr_nodes
  integer(int32), allocatable :: deg_target(:), deg(:), colind(:), src(:), deg4_nodes(:)
  integer(int64), allocatable :: rowptr(:), rev(:)
  real(real64) :: gamma, lmin, lmax, dl, lambda, tau, tau_sum, pend_frac
  integer :: nlambda, pend, pend_sum, ios
  character(len=256) :: arg, outfile

  integer, parameter :: kmin = 4
  real(real64), parameter :: cth = 0.5_real64

  ! defaults
  N       = 100000_int64
  gamma   = 3.5_real64
  nruns   = 5000_int64
  lmin    = 0.05_real64
  lmax    = 0.20_real64
  nlambda = 40
  outfile = 'part2_results.dat'

  call get_command_argument(1, arg)
  if (len_trim(arg) > 0) read(arg, *, iostat=ios) N
  call get_command_argument(2, arg)
  if (len_trim(arg) > 0) read(arg, *, iostat=ios) gamma
  call get_command_argument(3, arg)
  if (len_trim(arg) > 0) read(arg, *, iostat=ios) nruns
  call get_command_argument(4, arg)
  if (len_trim(arg) > 0) read(arg, *, iostat=ios) lmin
  call get_command_argument(5, arg)
  if (len_trim(arg) > 0) read(arg, *, iostat=ios) lmax
  call get_command_argument(6, arg)
  if (len_trim(arg) > 0) read(arg, *, iostat=ios) nlambda
  call get_command_argument(7, arg)
  if (len_trim(arg) > 0) outfile = trim(arg)

  call seed_rng()

  allocate(deg_target(N))
  call generate_powerlaw_degrees(N, gamma, kmin, deg_target)
  call build_erased_cm_csr(N, deg_target, deg, rowptr, colind, rev, src, M)
  deallocate(deg_target)

  nk4 = count(deg == kmin)
  if (nk4 == 0_int64) stop "No final degree-4 nodes found. Regenerate the network."

  allocate(deg4_nodes(nk4))
  r = 0
  do i = 1, N
    if (deg(i) == kmin) then
      r = r + 1
      deg4_nodes(r) = int(i, int32)
    endif
  enddo

  cov_thr_nodes = ceiling(cth * real(N, real64))

  open(unit=10, file=trim(outfile), status='replace', action='write')
  write(10,'(A)') '# lambda tau_non_endemic P_end N gamma nruns M'

  if (nlambda > 1) then
    dl = (lmax - lmin) / real(nlambda - 1, real64)
  else
    dl = 0.0_real64
  endif

  do i = 1, nlambda
    lambda = lmin + real(i - 1, real64) * dl
    tau_sum = 0.0_real64
    pend_sum = 0

    do r = 1, nruns
      call run_lifespan(lambda, cov_thr_nodes, &
                        deg4_nodes(1 + int(urand() * real(nk4, real64))), &
                        rowptr, colind, rev, src, tau, pend)
      pend_sum = pend_sum + pend
      if (pend == 0) tau_sum = tau_sum + tau
    enddo

    pend_frac = real(pend_sum, real64) / real(nruns, real64)

    if (nruns - pend_sum > 0) then
      tau_sum = tau_sum / real(nruns - pend_sum, real64)
    else
      tau_sum = 0.0_real64
    endif

    write(10,'(F12.6,1X,ES16.8,1X,F12.6,1X,I0,1X,F6.2,1X,I0,1X,I0)') &
         lambda, tau_sum, pend_frac, N, gamma, nruns, M
  enddo

  close(10)
end program sis_lifespan_cm