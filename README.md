# VARE(Visual Art Engine)

VARE는 유체 시뮬레이션을 위해 제작한 물리엔진입니다.
현재는 대규모 물 시뮬레이션을 시작으로 개발이 진행되고 있으며 주로 VFX 작업에 활용될 것으로 기대하고 있습니다.

VARE는 C++와 CUDA로 작성되었으며, Multithreading과 GPU기술을 기반으로 고속처리가 가능하도록 설계가 되어있습니다. 먼저 VFX아티스트들의 접근성을 고려해, 지금 버전은 Houdini의 플러그인으로 사용이 가능하도록 개발이 되었있으며 추후에는 Maya 아티스트 그리고 일반 사용자들이 보다 더 손쉽게 사용가능하도록 별도의 UI프로그램을 개발할 계획입니다.

VARE는 물의 사실적인 형태를 시뮬레이션하기 위해 기획된 프로젝트입니다. 한강 다리로 돌진하는 쓰나미를 한번 상상해 보십시오. 쓰나미는 다리에 부딫히면서 다양한 현상을 만들어 냅니다. 거대한 물덩어리가 도로위의 자동차를 덮치기도 하고 아주 작은 물방울들은 마치 연기 처럼 움직이기도 합니다. 이런 다양한 현상들을 물리적으로 모델링하기 위해 VARE는 만들어 졌으며, 다양한 기술들을 도입하고 있습니다.

아래 리스트는 VARE를 설계하기 위해 참조된 논문들입니다.
- Nielsen, Michael B., and Ole Østerby. "A two-continua approach to Eulerian simulation of water spray." ACM Transactions on Graphics (TOG) 32.4 (2013): 67.
- Zhu, Yongning, and Robert Bridson. "Animating sand as a fluid." ACM Transactions on Graphics (TOG) 24.3 (2005): 965-972.
- Bridson, Robert, Jim Houriham, and Marcus Nordenstam. "Curl-noise for procedural fluid flow." ACM Transactions on Graphics (ToG). Vol. 26. No. 3. ACM, 2007.
- Min, Chohong. "On reinitializing level set functions." Journal of computational physics 229.8 (2010): 2764-2772.
- Stomakhin, Alexey, et al. "A material point method for snow simulation." ACM Transactions on Graphics (TOG) 32.4 (2013): 102.
- Stam, Jos. "Stable Fluids." Siggraph. Vol. 99. 1999.
- Ihmsen, Markus, et al. "Unified spray, foam and air bubbles for particle-based fluids." The Visual Computer 28.6-8 (2012): 669-677.
Selle, Andrew, Nick Rasmussen, and Ronald Fedkiw. "A vortex particle method for smoke, water and explosions." ACM Transactions on Graphics (TOG). Vol. 24. No. 3. ACM, 2005.
