! matrix_ops.f90 - Operations matricielles creuses pour la reduction du connectome
!
! Cette bibliotheque Fortran implante les fonctions de reduction utilisees par
! FlyDrones (sensorimotor_core) pour comparer performance Fortran vs Python/NumPy.
!
! Compilation :
!   gfortran -shared -fPIC -O3 -o libmatrix_ops.so matrix_ops.f90
!
! Appele depuis Python via ctypes.

module matrix_ops
  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: c_reach, c_degree, c_sensorimotor_core_mask

contains

  subroutine c_reach(n, nnz, indptr, indices, seeds, n_seeds, hops, seen) bind(c, name="c_reach")
    integer(c_int), value, intent(in) :: n, nnz, n_seeds, hops
    integer(c_int64_t), intent(in) :: indptr(0:n), indices(0:nnz-1)
    integer(c_int64_t), intent(in) :: seeds(0:n_seeds-1)
    integer(c_int8_t), intent(out) :: seen(0:n-1)

    integer(c_int8_t), allocatable :: frontier(:), new_frontier(:)
    integer(c_int64_t) :: i, j, k, start, end_col, row
    integer(c_int) :: hop
    integer(c_int64_t) :: seed_idx

    allocate(frontier(0:n-1), new_frontier(0:n-1))

    seen = 0
    frontier = 0

    do seed_idx = 0, n_seeds-1
      i = seeds(seed_idx)
      if (i >= 0 .and. i < n) then
        frontier(i) = 1
        seen(i) = 1
      end if
    end do

    do hop = 1, hops
      new_frontier = 0

      do j = 0, n-1
        if (frontier(j) == 1) then
          start = indptr(j)
          end_col = indptr(j+1) - 1
          do k = start, end_col
            row = indices(k)
            new_frontier(row) = 1
          end do
        end if
      end do

      do i = 0, n-1
        if (new_frontier(i) == 1 .and. seen(i) == 0) then
          seen(i) = 1
        end if
      end do

      frontier = new_frontier

      if (sum(frontier) == 0) exit
    end do

    deallocate(frontier, new_frontier)
  end subroutine c_reach

  subroutine c_degree(n, nnz, indptr, data, degree) bind(c, name="c_degree")
    integer(c_int), value, intent(in) :: n, nnz
    integer(c_int64_t), intent(in) :: indptr(0:n)
    real(c_float), intent(in) :: data(0:nnz-1)
    real(c_float), intent(out) :: degree(0:n-1)

    integer(c_int64_t) :: j, k, start, end_col

    degree = 0.0_c_float

    do j = 0, n-1
      start = indptr(j)
      end_col = indptr(j+1) - 1
      do k = start, end_col
        degree(j) = degree(j) + abs(data(k))
      end do
    end do
  end subroutine c_degree

  subroutine c_sensorimotor_core_mask(n, nnz, indptr, indices, &
                                      inputs, n_inputs, &
                                      outputs, n_outputs, &
                                      hops, keep) bind(c, name="c_sensorimotor_core_mask")
    integer(c_int), value, intent(in) :: n, nnz, n_inputs, n_outputs, hops
    integer(c_int64_t), intent(in) :: indptr(0:n), indices(0:nnz-1)
    integer(c_int64_t), intent(in) :: inputs(0:n_inputs-1), outputs(0:n_outputs-1)
    integer(c_int8_t), intent(out) :: keep(0:n-1)

    integer(c_int8_t), allocatable :: down(:), up(:)
    integer(c_int64_t) :: i

    allocate(down(0:n-1), up(0:n-1))

    call c_reach(n, nnz, indptr, indices, inputs, n_inputs, hops, down)
    call c_reach_transpose(n, nnz, indptr, indices, outputs, n_outputs, hops, up)

    do i = 0, n-1
      if (down(i) == 1 .and. up(i) == 1) then
        keep(i) = 1
      else
        keep(i) = 0
      end if
    end do

    do i = 0, n_inputs-1
      if (inputs(i) >= 0 .and. inputs(i) < n) keep(inputs(i)) = 1
    end do
    do i = 0, n_outputs-1
      if (outputs(i) >= 0 .and. outputs(i) < n) keep(outputs(i)) = 1
    end do

    deallocate(down, up)
  end subroutine c_sensorimotor_core_mask

  subroutine c_reach_transpose(n, nnz, indptr, indices, seeds, n_seeds, hops, seen) bind(c, name="c_reach_transpose")
    integer(c_int), value, intent(in) :: n, nnz, n_seeds, hops
    integer(c_int64_t), intent(in) :: indptr(0:n), indices(0:nnz-1)
    integer(c_int64_t), intent(in) :: seeds(0:n_seeds-1)
    integer(c_int8_t), intent(out) :: seen(0:n-1)

    integer(c_int64_t), allocatable :: row_ptr(:), col_idx(:)
    integer(c_int8_t), allocatable :: frontier(:), new_frontier(:)
    integer(c_int64_t) :: i, j, k, col, row
    integer(c_int) :: hop
    integer(c_int64_t) :: seed_idx
    integer(c_int64_t), allocatable :: row_pos(:)

    allocate(row_ptr(0:n), col_idx(0:nnz-1))
    allocate(frontier(0:n-1), new_frontier(0:n-1))

    row_ptr = 0
    do j = 0, n-1
      do k = indptr(j), indptr(j+1) - 1
        row = indices(k)
        row_ptr(row+1) = row_ptr(row+1) + 1
      end do
    end do
    do i = 1, n
      row_ptr(i) = row_ptr(i) + row_ptr(i-1)
    end do

    allocate(row_pos(0:n-1))
    row_pos(0:n-1) = row_ptr(0:n-1)

    do j = 0, n-1
      do k = indptr(j), indptr(j+1) - 1
        row = indices(k)
        col_idx(row_pos(row)) = j
        row_pos(row) = row_pos(row) + 1
      end do
    end do
    deallocate(row_pos)

    seen = 0
    frontier = 0

    do seed_idx = 0, n_seeds-1
      i = seeds(seed_idx)
      if (i >= 0 .and. i < n) then
        frontier(i) = 1
        seen(i) = 1
      end if
    end do

    do hop = 1, hops
      new_frontier = 0

      do j = 0, n-1
        if (frontier(j) == 1) then
          do k = row_ptr(j), row_ptr(j+1) - 1
            col = col_idx(k)
            new_frontier(col) = 1
          end do
        end if
      end do

      do i = 0, n-1
        if (new_frontier(i) == 1 .and. seen(i) == 0) then
          seen(i) = 1
        end if
      end do

      frontier = new_frontier
      if (sum(frontier) == 0) exit
    end do

    deallocate(row_ptr, col_idx, frontier, new_frontier)
  end subroutine c_reach_transpose

end module matrix_ops
