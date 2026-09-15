cmake_minimum_required(VERSION 2.8.3)

project(PiPlus_S_12L8A0G2H0W_LSE_ZedMini_40V_260908)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

foreach(dir config launch meshes urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
