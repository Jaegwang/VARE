Name: Bora
Version: 1.1.0
Release: 1
Summary: BoraFX System RPM Package
License: Copyright 2018 by Dexter Studios
URL: http://dexterstudios.com
Group: Applications/Tools
Vendor: Dexter Studios
Packager: Wanho Choi (zelosdev@gmail.com)
Source0: %{name}-%{version}.tar.gz

%description
BoraFX is a C++ based in-house volume and particle simulation toolkit for Dexter Studios.
Authors: Jaegwang Lim, Julie Jang, and Wanho Choi

BuildRequires:
Requires:

###################################################################
%prep

cp %{_sourcedir}/%{name}-%{version}.tar.gz %{_builddir}
cd %{_builddir}
tar xvf %{name}-%{version}.tar.gz

###################################################################
%build

cd %{name}-%{version}
./doBuild

###################################################################
%clean
rm -rf $RPM_BUILD_ROOT

###################################################################
%install

# base
mkdir -p $RPM_BUILD_ROOT/netapp/backstage/pub/lib/bora/
cp -r %{_builddir}/%{name}-%{version}/base/header $RPM_BUILD_ROOT/netapp/backstage/pub/lib/bora/
mkdir -p $RPM_BUILD_ROOT/netapp/backstage/pub/lib/bora/lib/
cp %{_builddir}/%{name}-%{version}/build/lib/libBoraBase.so $RPM_BUILD_ROOT/netapp/backstage/pub/lib/bora/lib/

# renderman
# Use separate Git: rfm-extensions

# maya
mkdir -p $RPM_BUILD_ROOT/netapp/backstage/pub/lib/bora/maya/2017
cp -R %{_builddir}/%{name}-%{version}/build/maya/* $RPM_BUILD_ROOT/netapp/backstage/pub/lib/bora/maya/
#cp -R %{_builddir}/%{name}-%{version}/build/maya/* $RPM_BUILD_ROOT/netapp/backstage/pub/lib/bora/maya/

# houdini
mkdir -p $RPM_BUILD_ROOT/netapp/backstage/pub/apps/houdini2/tools/Bora
cp -R %{_builddir}/%{name}-%{version}/build/houdini/* $RPM_BUILD_ROOT/netapp/backstage/pub/apps/houdini2/tools/Bora/
#cp %{_builddir}/%{name}-%{version}/build/houdini/16.5.405/hpkg.ini $RPM_BUILD_ROOT/netapp/backstage/pub/apps/houdini2/tools/Bora/
#cp -r %{_builddir}/%{name}-%{version}/build/houdini/16.5.405/otls $RPM_BUILD_ROOT/netapp/backstage/pub/apps/houdini2/tools/Bora/
#cp -r %{_builddir}/%{name}-%{version}/build/houdini/16.5.405/plugins $RPM_BUILD_ROOT/netapp/backstage/pub/apps/houdini2/tools/Bora/

%files
/netapp/backstage/pub/lib/bora
/netapp/backstage/pub/apps/houdini2/tools/Bora

#############
## history ##

%changelog
* Wed Feb 13 2019 Jaegwang Lim - 1.1.0
- Removed: libgomp.so from BoraBase (due to Katana Problem)

* Mon Jan 28 2019 Jaegwang Lim - 1.0.1
- Added: BoraOcean Mesh Deformer

* Thu Dec 20 2018 Wanho Choi - 1.0.0
- Modified: BoraOcean - catrom bug fix
- Added: BoraFX for Houdini (FLIP Solver)

* Wed Sep 5 2018 Wanho Choi - 0.1.4
- Modified: BoraOcean - looplingDuration unit (from sec. to frames)
- Added: PxrBoraOcean - enableLoop

* Thu Apr 26 2018 Jaegwang Lim - 0.1.3
- Added: Secondary particles at FLIP solver

* Tue Apr 24 2018 Wanho Choi - 0.1.2
- Modified: BoraOcean - flowSpeed added
- Modified: PxrBoraOcean - rotationAngle added

* Thu Apr 12 2018 Jaegwang Lim - 0.1.1
- Added: FLIP HDA

* Mon Apr 02 2018 Wanho Choi - 0.1.0
- Initial RPM packaging

