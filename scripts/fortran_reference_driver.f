c     Reference driver for pqu7v2.f. The comparison script supplies one
c     fixed model, bypassing the original random-search main program.
      program kf_reference
      implicit none
      integer lpar, nsite, nsample
      parameter(lpar=16, nsite=1000, nsample=512)
      character*512 arg, filein, fileout
      integer i, ios, itpm, legge, nrnd, ns, nsp
      integer mdp(nsite), vint(nsite), vivt(nsite)
      real d2r, deg, eps, pig, residual
      real slat(nsite), slon(nsite), vlat(nsite), vlon(nsite)
      real vpar(lpar), vds(nsite), vkf(nsite), vkat(nsite), vkon(nsite)
      real re0v(nsite,2), the0v(nsite,2)
      real tv(nsample,nsite,2), yv(4,nsample,nsite,2)

      filein = ' '
      fileout = ' '
      call get_command_argument(1,filein)
      call get_command_argument(2,fileout)
      if (len_trim(filein).eq.0.or.len_trim(fileout).eq.0) stop 2

      vpar = 0.
      do i=1,16
        if (i.eq.9.or.i.eq.11.or.i.eq.13.or.i.eq.15) cycle
        call get_command_argument(i+2,arg)
        read(arg,*,iostat=ios) vpar(i)
        if (ios.ne.0) stop 3
      enddo
      call get_command_argument(11,arg)
      read(arg,*) nsp
      call get_command_argument(13,arg)
      read(arg,*) vpar(11)
      vpar(9) = real(nsp)
      if (nsp.gt.nsample) stop 4

      open(10,file=trim(filein),status='old')
      call readat(10,nsite,mdp,ns,vlat,vlon)
      close(10)
      pig=acos(-1.)
      deg=180.
      d2r=pig/deg
      eps=epsilon(1.)
      nrnd=1
      do i=1,ns
        slat(i)=vlat(i)*d2r
        slon(i)=vlon(i)*d2r
      enddo

c     iter=2 prevents the diagnostic-only writes used by the old main.
      open(20,status='scratch')
      call kfkern(2,2,-1,2,lpar,20,mdp,nrnd,ns,nsp,d2r,deg,eps,pig,
     . slat,slon,vint,vkat,vkon,vkf,vlat,vlon,vpar,re0v,the0v,tv,vds,
     . vivt,yv)
      close(20)

      residual=0.
      do i=1,ns
        residual=residual+real((vint(i)-mdp(i))*(vint(i)-mdp(i)))
      enddo
      open(30,file=trim(fileout),status='replace')
      write(30,'(A,1X,I0,1X,ES24.16)') '# summary',ns,residual
      do i=1,ns
        write(30,'(2(ES24.16,1X),2(I0,1X),2(ES24.16,1X))')
     .    vlon(i),vlat(i),mdp(i),vint(i),vds(i),vkf(i)
      enddo
      close(30)
      end
